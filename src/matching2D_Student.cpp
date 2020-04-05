
#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        vector<vector<cv::DMatch>> knn_matches;
        double t = (double) cv::getTickCount();
        matcher->knnMatch(descSource, descRef, knn_matches, 2);
        t = ((double) cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "KNN with n = " << knn_matches.size() << " matches in " << 1000*t/1.0 << " ms" << endl; 

        // Implement k-nearest-neighbor matching and filter matches using descriptor distance ratio test
        double minDescDistRatio = 0.8;
        for(auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {
            if((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }

        cout << "# keypoints removed = " << knn_matches.size() - matches.size() << endl;
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if(descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
    }
    else if(descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if(descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if(descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if(descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in image using the traditional Harris detector
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    int blockSize = 2;      // a blockSize x blockSize neighborhood for every pixel
    int apertureSize = 3;   // for sobel operator
    int minResponse = 100;  // minimum value for a corner in the 8-bit scaled response matrix
    double k = 0.04;        // Harris parameter

    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);

    double t = (double)cv::getTickCount();

    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // look for prominent corners and keypoints
    double maxOverlap = 0.0;
    for(size_t i = 0; i < dst_norm.rows; i++)
    {
        for(size_t j = 0; j < dst_norm.cols; j++)
        {
            int response = (int)dst_norm.at<float>(i,j);
            if(response > minResponse)
            {
                // only store points above a threshold
                cv::KeyPoint newKeypoint;
                newKeypoint.pt = cv::Point2f(j, i);
                newKeypoint.size = 2*apertureSize;
                newKeypoint.response = response;
                newKeypoint.class_id = 1;

                // perform non-maximal suppression in local neighbourhood around new key point
                bool bOverlap = false;
                for(auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeypoint, *it);
                    if(kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if(newKeypoint.response > (*it).response)
                        {
                            *it = newKeypoint;
                            break;
                        }
                    }
                }

                if(!bOverlap)
                {
                    keypoints.push_back(newKeypoint);
                }
            }
        }
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detector with n= " << keypoints.size() << " keypoints in " << 1000*t/1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = dst_norm_scaled.clone();
        cv::drawKeypoints(dst_norm_scaled, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 5);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, int detectorType, bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector;    
    string detectorName;

    switch (detectorType)
    {
        case 0:
        {
            // FAST
            detectorName = "FAST";
            int threshold = 30;
            bool nonMaxSuppression = true;

            cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
            detector = cv::FastFeatureDetector::create(threshold, nonMaxSuppression, type);
            break;
        }
        case 1:
        {
            // BRISK
            detectorName = "BRISK";
            detector = cv::BRISK::create();
            break;
        }
        case 2:
        {
            // ORB
            detectorName = "ORB";
            detector = cv::ORB::create();
            break;
        }
        case 3:
        {
            // AKAZE
            detectorName = "AKAZE";
            detector = cv::AKAZE::create();
            break;
        }
        case 4:
        {
            // SIFT
            detectorName = "SIFT";
            detector = cv::xfeatures2d::SIFT::create();
            break;
        }
    }

    double t = (double) cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
            
    cout << detectorName << " with n= " << keypoints.size() << " keypoints in " << 1000*t/1.0 << " ms" << endl;

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Modern Detector Results";
        cv::namedWindow(windowName, 2);
        imshow(windowName, visImage);

        cv::waitKey(0);
    }
}
