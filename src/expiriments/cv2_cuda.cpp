#include <iostream>
#include <unordered_map>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace cv::cuda;

void show(Mat img)
{
    imshow("img", img);
    waitKey(0);
}

void show(GpuMat img)
{
    Mat img_cpu;
    img.download(img_cpu);
    show(img_cpu);
}

void warpPiecewiseAffine2(InputArray _src, OutputArray _dst, std::vector<Point2f> &src_pts, std::vector<Point2f> &dst_pts)
{
    CV_Assert(src_pts.size() == dst_pts.size());

    class Hash
    {
    public:
        size_t operator()(const Point2f &p) const
        {
            return p.x + p.y;
        }
    };

    GpuMat src = _src.getGpuMat();
    Size size = src.size();

    Mat dstCpu(size, src.type(), Scalar(0, 0, 0));

    std::unordered_map<Point2f, Point2f, Hash>
        srcDstMapping;
    for (size_t i = 0; i < src_pts.size(); i++)
    {
        srcDstMapping[src_pts[i]] = dst_pts[i];
    }

    Subdiv2D subdiv = Subdiv2D(Rect(0, 0, size.width + 1, size.height + 1));
    subdiv.insert(src_pts);

    std::vector<cv::Vec6f> trianglesList;
    subdiv.getTriangleList(trianglesList);
    std::vector<Point> pt;
    for (cv::Vec6f triList : trianglesList)
    {
        Point2f from[3];
        Point2f to[3];
        from[0] = Point2f(triList[0], triList[1]);
        from[1] = Point2f(triList[2], triList[3]);
        from[2] = Point2f(triList[4], triList[5]);

        float minx = INT_MAX, miny = INT_MAX;
        float maxx = 0, maxy = 0;
        for (size_t i = 0; i < 3; i++)
        {
            Point2f pt = from[i];
            to[i] = srcDstMapping[pt];
            pt = to[i];
            maxx = round(cv::max(pt.x, maxx));
            maxy = round(cv::max(pt.y, maxy));
            minx = round(cv::min(pt.x, minx));
            miny = round(cv::min(pt.y, miny));
        }
        // skip if empty
        if (int(maxx) == int(minx) || int(maxy) == int(miny))
        {
            continue;
        }

        Mat m = getAffineTransform(from, to);

        GpuMat fulltransformed;
        cuda::warpAffine(src, fulltransformed, m, size);

        GpuMat transformedGpu = fulltransformed(Rect(minx, miny, maxx - minx, maxy - miny));
        Mat transformed;
        transformedGpu.download(transformed);

        std::vector<Point> toInt;
        for (auto pt : to)
        {
            toInt.push_back(Point(pt.x - minx, pt.y - miny));
        }

        Mat addTo = dstCpu.rowRange(miny, maxy).colRange(minx, maxx);
        Mat mask(transformed.size(), transformed.type(), Scalar(0, 0, 0));
        fillPoly(mask, toInt, Scalar(1, 1, 1));
        mask.setTo(0, addTo != 0);

        cv::multiply(transformed, mask, transformed);
        cv::add(addTo, transformed, addTo);
    }

    Mat srcCpu;
    src.download(srcCpu);

    srcCpu.copyTo(dstCpu, dstCpu == 0);

    _dst.create(size, src.type());
    GpuMat dst = _dst.getGpuMat();
    dst.upload(dstCpu);
}

int main(int argc, char const *argv[])
{
    using namespace std;
    std::vector<Point2f> from{Point2f(105, 137), Point2f(107, 147), Point2f(110, 156), Point2f(113, 165), Point2f(118, 173), Point2f(125, 180), Point2f(134, 185), Point2f(144, 188), Point2f(153, 188), Point2f(161, 185), Point2f(167, 180), Point2f(172, 173), Point2f(176, 165), Point2f(177, 156), Point2f(178, 147), Point2f(178, 138), Point2f(177, 128), Point2f(113, 129), Point2f(118, 124), Point2f(124, 122), Point2f(131, 122), Point2f(138, 124), Point2f(151, 123), Point2f(157, 119), Point2f(163, 117), Point2f(169, 117), Point2f(173, 121), Point2f(146, 132), Point2f(148, 139), Point2f(149, 145), Point2f(151, 152), Point2f(143, 157), Point2f(147, 157), Point2f(151, 158), Point2f(154, 157), Point2f(156, 155), Point2f(121, 136), Point2f(125, 133), Point2f(130, 133), Point2f(135, 135), Point2f(130, 137), Point2f(125, 138), Point2f(155, 133), Point2f(158, 129), Point2f(163, 128), Point2f(167, 129), Point2f(164, 132), Point2f(159, 133), Point2f(137, 168), Point2f(143, 166), Point2f(148, 165), Point2f(151, 165), Point2f(154, 164), Point2f(158, 164), Point2f(162, 165), Point2f(159, 169), Point2f(155, 170), Point2f(152, 171), Point2f(149, 171), Point2f(144, 170), Point2f(140, 168), Point2f(148, 167), Point2f(151, 167), Point2f(154, 166), Point2f(161, 165), Point2f(155, 166), Point2f(152, 167), Point2f(148, 167), Point2f(102, 115), Point2f(179, 115), Point2f(102, 191), Point2f(179, 191), Point2f(140, 115), Point2f(140, 191), Point2f(179, 153), Point2f(102, 153)};
    std::vector<Point2f> to{Point2f(105.0, 137.23225806451612), Point2f(107.0, 147.23225806451612), Point2f(110.0, 156.23225806451612), Point2f(113.0, 165.23225806451612), Point2f(118.0, 173.0), Point2f(125.0, 180.0), Point2f(134.0, 185.0), Point2f(144.0, 188.0), Point2f(153.0, 187.76774193548388), Point2f(161.0, 185.0), Point2f(167.0, 180.0), Point2f(172.0, 173.0), Point2f(176.0, 164.76774193548388), Point2f(177.0, 156.0), Point2f(178.0, 147.0), Point2f(178.0, 138.0), Point2f(177.0, 128.0), Point2f(113.0, 129.0), Point2f(118.0, 124.0), Point2f(124.0, 122.0), Point2f(131.0, 122.0), Point2f(138.0, 124.0), Point2f(151.0, 123.0), Point2f(157.0, 119.0), Point2f(163.0, 117.0), Point2f(169.0, 117.0), Point2f(173.0, 121.0), Point2f(146.0, 132.0), Point2f(148.0, 139.0), Point2f(149.0, 145.0), Point2f(151.0, 152.0), Point2f(143.0, 157.0), Point2f(147.0, 157.0), Point2f(151.0, 158.0), Point2f(154.0, 157.0), Point2f(156.0, 155.0), Point2f(121.0, 136.0), Point2f(125.0, 133.0), Point2f(130.0, 133.0), Point2f(135.0, 135.0), Point2f(130.0, 137.0), Point2f(125.0, 138.0), Point2f(155.0, 133.0), Point2f(158.0, 129.0), Point2f(163.0, 128.0), Point2f(167.0, 129.0), Point2f(164.0, 132.0), Point2f(159.0, 133.0), Point2f(136.76699029126215, 168.0), Point2f(143.0, 166.0), Point2f(148.0, 165.0), Point2f(151.0, 165.0), Point2f(154.0, 164.0), Point2f(158.0, 164.0), Point2f(162.0, 165.0), Point2f(159.0, 169.0), Point2f(155.0, 170.0), Point2f(152.0, 171.0), Point2f(149.0, 171.0), Point2f(144.0, 170.0), Point2f(140.0, 168.0), Point2f(148.0, 167.0), Point2f(151.0, 167.0), Point2f(154.0, 166.0), Point2f(161.0, 165.0), Point2f(155.0, 166.0), Point2f(152.0, 167.0), Point2f(148.0, 167.0), Point2f(102, 115), Point2f(179, 115), Point2f(102, 191), Point2f(179, 191), Point2f(140, 115), Point2f(140, 191), Point2f(179, 153), Point2f(102, 153)};

    Mat img = imread("image2.jpg");
    GpuMat img_gpu;
    img_gpu.upload(img);

    auto start = chrono::high_resolution_clock::now();
    GpuMat out;
    long x = strtol(argv[1], NULL, 10);
    for (int i = 0; i < x; i++)
    {
        warpPiecewiseAffine2(img_gpu, out, from, to);
    }
    auto end = chrono::high_resolution_clock::now();
    cout << (end - start).count() / 1e6 << "ms" << endl;
    // show(img);
    // show(out);
    return 0;
}
