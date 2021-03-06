
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    for(const auto& match : kptMatches) {
        const auto &currKeyPoint = kptsCurr[match.trainIdx].pt;
        if (boundingBox.roi.contains(currKeyPoint)) {
            boundingBox.kptMatches.emplace_back(match);
        }
    }

    double sum = 0;
    // Remove outlier matches based on the euclidean distance between them in relation to all the matches in the bounding box.
    for (const auto& it : boundingBox.kptMatches) 
    {
        cv::KeyPoint kpCurr = kptsCurr.at(it.trainIdx);
        cv::KeyPoint kpPrev = kptsPrev.at(it.queryIdx);
        double dist = cv::norm(kpCurr.pt - kpPrev.pt);
        sum += dist;
    }
    cout << std::endl;
    double mean = sum / boundingBox.kptMatches.size();

    constexpr double ratio = 1.5;
    for (auto it = boundingBox.kptMatches.begin(); it < boundingBox.kptMatches.end();) {
        cv::KeyPoint kpCurr = kptsCurr.at(it->trainIdx);
        cv::KeyPoint kpPrev = kptsPrev.at(it->queryIdx);
        double dist = cv::norm(kpCurr.pt - kpPrev.pt);

        if (dist >= mean * ratio) 
        {
            boundingBox.kptMatches.erase(it);
        }
        else 
        {
            it++;
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; it1++) 
    {

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); it2++) 
        {

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > numeric_limits<double>::epsilon() && distCurr >= minDist) 
            {
                // avoid division by zero
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        }
    }

    
    if (distRatios.empty()) 
    {
        TTC = NAN;
        return;
    }

    sort(distRatios.begin(), distRatios.end());


    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex];

    cout << "medDistRatio = " << medDistRatio << endl;

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double dT = 1 / frameRate;
    constexpr double laneWidth = 4.0;
    constexpr float clusterTolerance = 0.1;

    // find closest distance to LiDAR points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;

    cout << "Process previous frame..." << endl;
    vector<LidarPoint> lidarPointsPrevClustered = removeLidarOutlier(lidarPointsPrev, clusterTolerance);

    cout << "Process current frame..." << endl;
    vector<LidarPoint> lidarPointsCurrClustered = removeLidarOutlier(lidarPointsCurr, clusterTolerance);


    for (const auto & it : lidarPointsPrevClustered) 
    {
        if (abs(it.y) <= laneWidth / 2.0) 
        {
            minXPrev = it.x < minXPrev ? it.x : minXPrev;
        }
    }

    for (const auto & it : lidarPointsCurrClustered) 
    {
        if (abs(it.y) <= laneWidth / 2.0) 
        {
            minXCurr = it.x < minXCurr ? it.x : minXCurr;
        }
    }

    cout << "Prev min X = " << minXPrev << endl;
    cout << "Curr min X = " << minXCurr << endl;

    TTC = minXCurr * dT / (minXPrev - minXCurr);
}


template<typename KeyType, typename ValueType>
std::pair<KeyType, ValueType> get_max(const std::map<KeyType, ValueType>& x) 
{
    using pairtype = std::pair<KeyType, ValueType>;
    return *std::max_element(x.begin(), x.end(), [] (const pairtype & p1, const pairtype & p2)
    {
        return p1.second < p2.second;
    });
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    for (const auto& prevBox : prevFrame.boundingBoxes) 
    {
        map<int, int> m;
        for (const auto& currBox : currFrame.boundingBoxes) 
        {
            for (const auto &match : matches) 
            {
                const auto &prevKeyPoint = prevFrame.keypoints[match.queryIdx].pt;
                if (prevBox.roi.contains(prevKeyPoint)) 
                {
                    const auto &currKeyPoint = currFrame.keypoints[match.trainIdx].pt;
                    if (currBox.roi.contains(currKeyPoint)) 
                    {
                        if(0 == m.count(currBox.boxID)) 
                        {
                            m[currBox.boxID] = 1;
                        }
                        else 
                        {
                            m[currBox.boxID]++;
                        }
                    }
                }
            }
        }

        auto max=get_max(m);

        bbBestMatches[prevBox.boxID] = max.first;
        cout << "ID Matching: " << prevBox.boxID << "=>" << max.first << endl;
    }
}


std::vector<LidarPoint> removeLidarOutlier(const std::vector<LidarPoint> &lidarPoints, float clusterTolerance) 
{
    auto treePrev = make_shared<KdTree>();
    vector<vector<float>> points;
    for (int i=0; i< lidarPoints.size(); i++) 
    {
        vector<float> point({
            static_cast<float>(lidarPoints[i].x),               
            static_cast<float>(lidarPoints[i].y),
            static_cast<float>(lidarPoints[i].z)
        });

        points.push_back(point);
        treePrev->insert(points[i], i);
    }
    vector<vector<int>> cluster_indices = euclideanCluster(points, treePrev, clusterTolerance);

    vector<LidarPoint> maxLidarPointsCluster;
    for (const auto& get_indices : cluster_indices) 
    {
        vector<LidarPoint> temp;
        for (const auto index : get_indices) 
        {
            temp.push_back(lidarPoints[index]);
        }

        cout << "Cluster size = " << temp.size() << std::endl;

        if (temp.size() > maxLidarPointsCluster.size()) 
        {
            maxLidarPointsCluster = std::move(temp);
        }
    }

    cout << "Max cluster size = " << maxLidarPointsCluster.size() << std::endl;
    return maxLidarPointsCluster;
}


void clusterHelper(int index, const std::vector<std::vector<float>>& points, std::vector<int>& cluster, std::vector<bool>& processed,
        const std::shared_ptr<KdTree>& tree, float distanceTol) {
    processed[index] = true;
    cluster.push_back(index);

    vector<int> nearest = tree->search(points[index], distanceTol);

    for (int id : nearest) 
    {
        if (!processed[id]) 
        {
            clusterHelper(id, points, cluster, processed, tree, distanceTol);
        }
    }
}


std::vector<std::vector<int>> euclideanCluster(const std::vector<std::vector<float>>& points,
        const std::shared_ptr<KdTree>& tree, float distanceTol) 
{

    vector<vector<int>> clusters;
    vector<bool> processed(points.size(), false);

    int i = 0;
    while (i < points.size()) 
    {
        if (processed[i]) 
        {
            i++;
            continue;
        }

        vector<int> cluster;
        clusterHelper(i, points, cluster, processed, tree, distanceTol);
        clusters.push_back(cluster);
        i++;
    }

    return clusters;
}
