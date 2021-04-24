//
//  main.cpp
//  opencv
//
//  Created by apple on 2021/3/20.
//
/*
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char** argv)
{
    Mat image1, image2;
    image1 = imread("/Users/apple/Desktop/00.jpeg");
    if (image1.data == 0 )
    {
        printf("could not load image...");
        return -1;
    }
    namedWindow("input",WINDOW_AUTOSIZE);
    imshow("input", image1);

    int width = (image1.cols-1) * image1.channels();
    int height = image1.rows-1;
    int first = image1.channels();
    image2 = Mat::zeros(image1.size(), image1.type());
    for (int row = 1; row < height; row++)
    {
        const uchar* current = image1.ptr(row);
        const uchar* previous = image1.ptr(row-1);
        const uchar* next = image1.ptr(row+1);
        uchar* output = image2.ptr(row);
        for(int col = first; col < width; col++)
        {
            output[col] = saturate_cast<uchar>(5 * current[col] - (current[col - first] + current[col + first] + previous[col] + next[col]));
        }
    }
    double t = getTickCount();
    Mat kernel = (Mat_<char> (3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0 );
    filter2D(image1, image2, image1.depth(), kernel);
    double timeconsume = (getTickCount()-t)/getTickFrequency();
    printf("time consume %.2f\n", timeconsume);
    namedWindow("output",WINDOW_AUTOSIZE);
    imshow("output", image2);
    
    waitKey(0);
    return 0;
}

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    Mat image1, image2;
    image1 = imread("/Users/apple/Desktop/00.jpeg");
    if(!image1.data)
    {
        printf("could not load image...");
        return -1;
    }
    namedWindow("input",WINDOW_AUTOSIZE);
    imshow("input",image1);
    
    image2 = Mat(image1.size(), image1.type());
    image2 = Scalar(127, 0, 255);
    namedWindow("output",WINDOW_AUTOSIZE);
    imshow("output",image2);
    
    image2 = image1.clone();
    namedWindow("output",WINDOW_AUTOSIZE);
    imshow("output",image2);
    
    //image1.copyTo(image2);
    cvtColor(image1, image2, COLOR_BGR2GRAY);
    printf("input image channnels:%d\n",image1.channels());
    printf("output image channnels:%d\n",image2.channels());
    namedWindow("output",WINDOW_AUTOSIZE);
    imshow("output",image2);
    
    int height = image2.rows;
    int width = image2.cols;
    printf("the height of image2 is: %d\n", height);
    printf("the width of image2 is: %d\n", width);
    
    const uchar* firstrow = image1.ptr();
    printf("first pixel value : %d", *firstrow);
    
    Mat M(100,100,CV_8UC3,Scalar(0,0,255));
    //cout  << "M = "  << endl << M << endl;
    imshow("M = ", M);
    
    Mat m1;
    m1.create(image1.size(), image1.type());
    m1 = Scalar(0, 0, 255);
    imshow("m1", m1);
    
    Mat m2 = Mat::zeros(image1.size(),image1.type());
    imshow("m2 = ", m2);
    waitKey(0);
    return 0;
}

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    Mat image1, image2;
    image1 = imread("/Users/apple/Desktop/00.jpeg");
    if(!image1.data)
    {
        printf("could not load image...");
        return -1;
    }
    namedWindow("image1", WINDOW_AUTOSIZE);
    imshow("image1",image1);
    cvtColor(image1, image2, COLOR_BGR2GRAY);
    //namedWindow("image2",WINDOW_AUTOSIZE);
    //imshow("image2", image2);
    
    int height = image2.rows;
    int width = image2.cols;
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            int pixel = image2.at<uchar>(row, col);
            image2.at<uchar>(row, col) =  255 - pixel;
        }
    }
    //imshow("changed image", image2);
    
    Mat image3;
    image3.create(image1.size(), image1.type());
    height = image1.rows;
    width = image1.cols;
    int first = image1.channels();
    
    for (int row = 0; row < height; row++)
    {
        for (int col = first; col < width; col++)
        {
            if(first == 1)
            {
                int pixel = image2.at<uchar>(row, col);
                image2.at<uchar>(row, col) =  255 - pixel;
            }
            else if(first == 3)
            {
                int b = image1.at<Vec3b>(row, col)[0];
                int g = image1.at<Vec3b>(row, col)[1];
                int r = image1.at<Vec3b>(row, col)[2];
                image3.at<Vec3b>(row, col)[0] = 0;
                image3.at<Vec3b>(row, col)[1] = g;
                image3.at<Vec3b>(row, col)[2] = r;
                
                image2.at<uchar>(row, col) = min(r, min(b,g));
            }
        }
    }
    //bitwise_not(image1, image3);
    //imshow("image3", image3);
    imshow("image2", image2);
    waitKey(0);
    return 0;
}

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    Mat image1, image2, dst;
    double alpha = 0.5;
    image1 = imread("/Users/apple/Desktop/00.jpeg");
    image2 = imread("/Users/apple/Desktop/11.jpeg");
    if(!image1.data)
    {
        cout << "could not load image 00..." << endl;
        return -1;
    }
    if(!image2.data)
    {
        cout << "could not load image 11..." << endl;
        return -1;
    }
    if(image1.type() == image2.type() && image1.rows == image2.rows && image1.cols == image2.cols)
    {
        addWeighted(image1, alpha, image2, (1 - alpha), 0, dst);
        //multiply(image1, image2, dst);
        imshow("image1", image1);
        imshow("image2", image2);
        imshow("output", dst);
    }
    else
    {
        printf("could not blend images...");
        return -1;
    }
    waitKey(0);
    return 0;
}
 
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    Mat image1, image2, image3;
    image1 = imread("/Users/apple/Desktop/00.jpeg");
    if(!image1.data)
    {
        cout << "could not load image 00..." << endl;
        return -1;
    }
   
    cvtColor(image1, image1, COLOR_BGR2GRAY);
    imshow("image1", image1);
    int height = image1.rows;
    int width = image1.cols;
    float alpha = 1.2;
    float beta = 30;
    image1.convertTo(image3, CV_32F);
    image2 = Mat::zeros(image1.size(), image1.type());
    for(int row = 0; row < height; row++)
    {
        for(int col = 0; col < width; col++)
        {
            if(image1.channels() == 3)
            {
                float b = image3.at<Vec3f>(row, col)[0];
                float g = image3.at<Vec3f>(row, col)[1];
                float r = image3.at<Vec3f>(row, col)[2];
                image2.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(b * alpha + beta);
                image2.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(g * alpha + beta);
                image2.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(r * alpha + beta);
            }
            if(image1.channels() == 1)
            {
                float v = image1.at<uchar>(row, col);
                image2.at<uchar>(row, col) = saturate_cast<uchar>(v * alpha + beta);
            }
        }
    }
    imshow("image2", image2);
    waitKey(0);
    return 0;
}

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat image1;

void Myline();
void MyRectangle();
void MyEllipse();
void MyCircle();
void MyPolygon();
void Randomline();

int main(int argc, char** argv)
{
    image1 = imread("/Users/apple/Desktop/11.jpeg");
    if(image1.data == 0)
    {
        cout << "could not load image..." << endl;
        return -1;
    }
    Myline();
    MyRectangle();
    MyEllipse();
    MyCircle();
    MyPolygon();
    putText(image1, "hello opencv", Point(300, 300), QT_FONT_BLACK, 2.0, Scalar(12, 23, 200), 3, 8);
    imshow("image1",image1);
    Randomline();
    waitKey(0);
    return 0;
}

void Myline()
{
    Point p1 = Point(20,30);
    Point p2 = Point(400, 400);
    Scalar color = Scalar(0,0,255);
    line(image1, p1, p2, color, 2, LINE_AA);
}

void MyRectangle()
{
    Rect r1 = Rect(200, 100, 300, 300);
    Scalar color = Scalar(255, 0, 0);
    rectangle(image1, r1, color, 2, LINE_8);
}

void MyEllipse()
{
    Scalar color = Scalar(0, 255, 0);
    ellipse(image1, Point(image1.cols/2, image1.rows/2), Size(image1.cols/4,image1.rows/8), 90, 0, 360, color, 2, LINE_8);
}

void MyCircle()
{
    Scalar color = Scalar(0, 255, 255);
    Point center = Point(image1.cols/2, image1.rows/2);
    circle(image1, center, 200, color, 2, LINE_8);
}

void MyPolygon()
{
    Point p[1][4];
    p[0][0] = Point(100,100);
    p[0][1] = Point(100,200);
    p[0][2] = Point(200,200);
    p[0][3] = Point(200,100);

    const Point* pp[] = {p[0]};
    int np[] = {4};
    Scalar color = Scalar(255, 0, 255);
    
    fillPoly(image1, pp, np, 1, color);
}

void Randomline()
{
    RNG rng;
    Point p1, p2;
    Mat image2 = Mat::zeros(image1.size(), image1.type());
    for (int i = 0; i < 10000; i++)
    {
    p1.x = rng.uniform(0, image1.cols);
    p2.x = rng.uniform(0, image1.cols);
    p1.y = rng.uniform(0, image1.rows);
    p2.y = rng.uniform(0, image1.rows);
    Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    if (waitKey(50) > 0)
    {
        break;
    }
    line(image2, p1, p2, color, 1, 8);
    imshow("image2", image2);
    }
}

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
    
int main(int argc, char** argv)
{
    Mat image1, image2, image3;
    image1 = imread("/Users/apple/Desktop/00.jpeg");
    if(!image1.data)
    {
        cout << "could not load image..." << endl;
        return -1;
    }
    imshow("image1", image1);
    
    blur(image1, image2, Size(11, 11), Point(-1, -1));
    imshow("image2", image2);
    
    GaussianBlur(image1, image3, Size(11, 11), 11, 11);
    imshow("image3", image3);
    
    waitKey(0);
    return  0;
}
    
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    Mat image1, image2, image3;
    image1 = imread("/Users/apple/Desktop/11.jpeg");
    if(!image1.data)
    {
        cout << "could not load image..." << endl;
        return -1;
    }
    imshow("image1", image1);
    
    //medianBlur(image1, image2, 3);
    bilateralFilter(image1, image2, 50, 30, 500);
    imshow("image2", image2);
    
    Mat kernel = (Mat_<int>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    filter2D(image2, image3, -1, kernel, Point(-1, -1), 0);
    imshow("image3", image3);
    
    waitKey(0);
    return 0;
}

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat image1, image2;
int element_size = 3;
int max_size = 21;
char output[] = "image2";
void CallBack(int, void*);

int main(int argc, char** argv)
{
   
    image1 = imread("/Users/apple/Desktop/00.jpeg");
    if(image1.data == 0)
    {
        cout << "could not load image..." << endl;
        return -1;
    }
    cvtColor(image1, image1, COLOR_BGR2GRAY);
    imshow("image1", image1);
    
    namedWindow(output, WINDOW_AUTOSIZE);
    createTrackbar("value:", output, &element_size, max_size, CallBack);
    CallBack(0, 0);
    
    waitKey(0);
    return 0;
}

void CallBack(int, void*)
{
    int s = element_size * 3;
    Mat se = getStructuringElement(MORPH_RECT, Size(s, s), Point(-1, -1));
    //dilate(image1, image2, se, Point(-1, -1), 1);
    erode(image1, image2, se, Point(-1, -1), 1);
    imshow(output, image2);
}

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    Mat image1,image2;
    image1 = imread("/Users/apple/Desktop/00.jpeg");
    if(!image1.data)
    {
        cout << "could not load image..." << endl;
        return -1;
    }
    cvtColor(image1, image1, COLOR_BGR2GRAY);
    imshow("image1", image1);
    
    Mat kernel = getStructuringElement(MORPH_RECT, Size(11, 11), Point(-1,-1));
    morphologyEx(image1, image2, MORPH_TOPHAT, kernel);
    imshow("image2", image2);
    
    waitKey(0);
    return 0;
}

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    Mat image1, image2, image3, image4, image5;
    image1 = imread("/Users/apple/Desktop/000.jpeg");
    if(!image1.data)
    {
        cout << "could not load image..." << endl;
        return -1;
    }
    cvtColor(image1, image2, COLOR_RGB2GRAY);
    imshow("image2", image2);
    adaptiveThreshold(~image2, image3, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 15, -2);
    imshow("image3", image3);
     
    Mat hline = getStructuringElement(MORPH_RECT, Size(image1.cols/16, 1), Point(-1, -1));
    Mat vline = getStructuringElement(MORPH_RECT, Size(1, image1.rows/16), Point(-1, -1));
    
    //erode(image3, image4, vline);
    //dilate(image4, image5, vline);
    morphologyEx(image3, image4, MORPH_OPEN, vline);
    bitwise_not(image4, image4);
    imshow("image4", image4);
    
    waitKey(0);
    return 0;
}

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    Mat image1, image2, image3, image4, image5, image6, dogimage;
    image1 = imread("/Users/apple/Desktop/00.jpeg");
    if(!image1.data)
    {
        cout << "could not load image..." << endl;
        return -1;
    }
    imshow("image1", image1);
    
    pyrUp(image1, image2, Size(image1.cols * 2, image1.rows * 2));
    imshow("image2", image2);
    
    pyrDown(image1, image3, Size(image1.cols / 2, image1.rows / 2));
    imshow("image3", image3);
    
    //cvtColor(image1, image4, COLOR_BGR2GRAY);
    GaussianBlur(image1, image5, Size(3, 3), 0, 0);
    GaussianBlur(image5, image6, Size(3, 3), 0, 0);
    subtract(image5, image6, dogimage);
    normalize(dogimage, dogimage, 255, 0, NORM_MINMAX);
    imshow("dogimage", dogimage);
    
    waitKey(0);
    return 0;
}

#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

Mat image1, image2, image3;
int threshold_value = 127;
int threshold_max = 255;
const char output[] = "image3";
void Threshold_demo(int, void*);

int type_value = 2;
int type_max = 4;


int main(int argc, char** argv)
{
    image1 = imread("/Users/apple/Desktop/00.jpeg");
    if(image1.data == 0)
    {
        cout << "could not load image..." << endl;
        return -1;
    }
    imshow("image1", image1);
    namedWindow(output, WINDOW_AUTOSIZE);
    createTrackbar("Threshold value:", output, &threshold_value, threshold_max, Threshold_demo);
    createTrackbar("Type value:", output, &type_value, type_max, Threshold_demo);
    Threshold_demo(0, 0);
    waitKey(0);
    return 0;
}

void Threshold_demo(int, void*)
{
    cvtColor(image1, image2, COLOR_BGR2GRAY);
    threshold(image2, image3, 0, 255, THRESH_TRIANGLE | type_value);
    imshow(output, image3);
}

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    Mat image1, image2, image3, image4;
    int ksize = 0;
    int i = 0;
    image1 = imread("/Users/apple/Desktop/00.jpeg");
    if(!image1.data)
    {
        cout << "could not load image..." << endl;
        return -1;
    }
    imshow("image1", image1);
    
    Mat kernel_x = (Mat_<int>(3, 3) << -1, 0, 1 , -2, 0, 2, -1, 0, 1);
    filter2D(image1, image2, -1, kernel_x, Point(-1, -1), 0);
    imshow("image2", image2);
    
    Mat kernel_y = (Mat_<int>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    filter2D(image1, image3, -1, kernel_y, Point(-1, -1), 0);
    imshow("image3", image3);
    
    Mat kernel = (Mat_<int>(3, 3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);
    filter2D(image1, image4, -1, kernel, Point(-1, -1), 0);
    imshow("image4", image4);
    while(true)
    {
        if(waitKey(500) == 27)
        {
            break;
        }
        ksize = (i % 8) * 2 + 5;
        Mat kernel = Mat::ones(Size(ksize, ksize), CV_32F)/ (ksize * ksize);
        filter2D(image1, image2, -1, kernel, Point(-1, -1), 0);
        i++;
        imshow("image2", image2);
    }
    return 0;
}

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    Mat image1, image2;
    image1 = imread("/Users/apple/Desktop/00.jpeg");
    if(!image1.data)
    {
        cout << "could not load image..." << endl;
        return -1;
    }
    imshow("image1", image1);
    
    int top = (int)(0.05 * image1.rows);
    int bottom = (int)(0.05 * image1.rows);
    int left = (int)(0.05 * image1.cols);
    int right = (int)(0.05 * image1.cols);
    RNG rng;
    int bordertype = BORDER_DEFAULT;
    char c = 0;
    while(c != 27)
    {
        c = waitKey(500);
        if(c == 'r')
        {
            bordertype = BORDER_REPLICATE;
        }
        else if (c == 'v')
        {
            bordertype = BORDER_WRAP;
        }
        else if (c == 'c')
        {
            bordertype = BORDER_CONSTANT;
        }
        else
        {
            bordertype = BORDER_DEFAULT;
        }
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        copyMakeBorder(image1, image2, top, bottom, left, right, bordertype, color);
        imshow("image2", image2);
    }
    GaussianBlur(image1, image2, Size(11, 11), 0, 0, BORDER_WRAP);
    imshow("image2", image2);
    waitKey(0);
    return 0;
}

#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    Mat image1, image2, image3, image4, image5, image6;
    image1 = imread("/Users/apple/Desktop/00.jpeg");
    if(!image1.data)
    {
        cout << "could not load image..." << endl;
        return -1;
    }
    imshow("image1", image1);
    GaussianBlur(image1, image2, Size(3, 3), 0, 0);
    cvtColor(image2, image3, COLOR_BGR2GRAY);
    imshow("image3", image3);
    
    Sobel(image3, image4, CV_16S, 1, 0, 3);
    Sobel(image3, image5, CV_16S, 0, 1, 3);
    
    //Scharr(image3, image4, CV_16S, 1, 0, 3);
    //Scharr(image3, image5, CV_16S, 0, 1, 3);
    
    convertScaleAbs(image4, image4);
    convertScaleAbs(image5, image5);
    imshow("image4", image4);
    imshow("image5", image5);
    
    //addWeighted(image4, 0.5, image5, 0.5, 0, image6);
    //bitwise_not(image6, image6);
    image6 = Mat(image4.size(), image4.type());
    int width =image4.cols;
    int height = image4.rows;
    for(int row = 0; row < height; row++)
    {
        for(int col = 0; col < width; col++)
        {
            int x = image4.at<uchar>(row, col);
            int y = image5.at<uchar>(row, col);
            int xy = sqrt(pow(x, 2) + pow(y, 2));
            image6.at<uchar>(row, col) = saturate_cast<uchar>(xy);
        }
    }
    //bitwise_not(image6, image6);
    imshow("image6", image6);
    
    waitKey(0);
    return 0;
}

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    Mat image1, image2, image3, image4;
    image1 = imread("/Users/apple/Desktop/00.jpeg");
    if(!image1.data)
    {
        cout << "could not load image..." << endl;
        return -1;
    }
    imshow("image1", image1);
    GaussianBlur(image1, image2, Size(3, 3), 0, 0);
    cvtColor(image2, image3, COLOR_RGB2GRAY);
    Laplacian(image3, image4, CV_16S, 3);
    convertScaleAbs(image4, image4);
    threshold(image4, image4, 0, 255, THRESH_OTSU | THRESH_BINARY);
    
    imshow("image4", image4);
    
    waitKey(0);
    return 0;
}

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat image1, image2, image3, image4;
int t1 = 50;
int max_value = 255;
void canny_demo(int, void*);
char output[] = "result";

int main(int argc, char** argv)
{
    image1 = imread("/Users/apple/Desktop/00.jpeg");
    if(!image1.data)
    {
        cout << "could not load image..." << endl;
        return -1;
    }
    imshow("image1", image1);
    namedWindow(output, WINDOW_AUTOSIZE);
    createTrackbar("Threshold value:", output, &t1, max_value, canny_demo);
    
    cvtColor(image1, image2, COLOR_RGB2GRAY);
    canny_demo(0, 0);
    waitKey(0);
    return 0;
}

void canny_demo(int, void*)
{
    blur(image2, image2, Size(3, 3), Point(-1, -1), BORDER_DEFAULT);
    Canny(image2, image3, t1, t1 * 2, 3, false);
    
    image4.create(image1.size(), image1.type());
    image1.copyTo(image4, image3);
    imshow(output, image4);
}

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    Mat image1, image2, image3;
    image1 = imread("/Users/apple/Desktop/111.jpeg");
    if(!image1.data)
    {
        cout << "could not load image..." << endl;
        return -1;
    }
    imshow("image1", image1);
    
    Canny(image1, image2, 150, 200);
    cvtColor(image2, image3, COLOR_GRAY2BGR);
    imshow("image2", image2);
    
    vector<Vec4f> plines;
    HoughLinesP(image2, plines, 1, CV_PI / 180.0, 10, 0, 10);
    Scalar color = Scalar(0, 255 ,255);
    for(size_t i = 0; i < plines.size(); i++)
    {
        Vec4f hline = plines[i];
        line(image3, Point(hline[0], hline[1]), Point(hline[2], hline[3]), color, 3, LINE_AA);
    }
    imshow("image3", image3);
    waitKey(0);
    return 0;
}

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    Mat image1, image2, image3;
    image1 = imread("/Users/apple/Desktop/shit.jpeg");
    if(!image1.data)
    {
        cout << "could not load image..." << endl;
        return -1;
    }
    imshow("image1", image1);
    
    medianBlur(image1, image2, 3);
    cvtColor(image2, image2, COLOR_BGR2GRAY);
    
    vector<Vec3f> pcircles;
    HoughCircles(image2, pcircles, HOUGH_GRADIENT, 1, 10, 100, 40, 5, 50);
    
    image1.copyTo(image3);
    for(size_t i = 0; i < pcircles.size(); i++)
    {
        Vec3f center = pcircles[i];
        circle(image3, Point(center[0], center[1]), center[2], Scalar(0, 255, 0), 2, LINE_AA);
        circle(image3, Point(center[0], center[1]), 2, Scalar(0, 255, 0), 2, LINE_AA);
    }
    imshow("image3", image3);
    waitKey(0);
    return 0;
}

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat image1, image2, map_x, map_y;
int in = 0;
void update();

int main(int argc, char** argv)
{
    image1 = imread("/Users/apple/Desktop/00.jpeg");
    if(!image1.data)
    {
        cout << "could not load image..." << endl;
        return -1;
    }
    imshow("image1", image1);
    
    map_x.create(image1.size(), CV_32FC1);
    map_y.create(image1.size(), CV_32FC1);
    
    while(true)
    {
        char c = waitKey(500);
        in = c % 4;
        if(c == 27)
        {
            break;
        }
        update();
        remap(image1, image2, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 255, 255));
        imshow("image2", image2);
    }
    
    
    waitKey(0);
    return 0;
}

void update()
{
    for(int row = 0; row < image1.rows ; row++)
    {
        for(int col = 0; col < image1.cols; col++)
        {
            switch(in)
            {
                case 0:
                    if(col > (image1.cols * 0.25) && col < (image1.cols * 0.75) && row > (image1.rows * 0.25) && row < (image1.rows * 0.75))
                    {
                        map_x.at<float>(row, col) = 2 * (col - (image1.cols * 0.25) + 0.5);
                        map_y.at<float>(row, col) = 2 * (row - (image1.rows * 0.25) + 0.5);
                    }
                    else
                    {
                        map_x.at<float>(row, col) = 0;
                        map_y.at<float>(row, col) = 0;
                    }
                    break;
                case 1:
                    map_x.at<float>(row, col) = image1.cols - col - 1;
                    map_y.at<float>(row, col) = row;
                    break;
                case 2:
                    map_x.at<float>(row, col) = col;
                    map_y.at<float>(row, col) = image1.rows - row - 1;
                    break;
                case 3:
                    map_x.at<float>(row, col) = image1.cols - col - 1;
                    map_y.at<float>(row, col) = image1.rows - row - 1;
                    break;
            }
        }
    }
}

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char**)
{
    Mat image1, image2;
    image1 = imread("/Users/apple/Desktop/00.jpeg");
    if(!image1.data)
    {
        cout << "could not load image..." << endl;
        return -1;
    }
    imshow("image1", image1);
    cvtColor(image1, image2, COLOR_BGR2GRAY);
    equalizeHist(image2, image2);
    imshow("image2", image2);
    waitKey(0);
    return 0;
}

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    Mat image1;
    image1 = imread("/Users/apple/Desktop/00.jpeg");
    if(!image1.data)
    {
        cout << "could not load image..." << endl;
        return -1;
    }
    imshow("image1", image1);
    
    vector<Mat> bgr_planes;
    Mat b_hist, g_hist, r_hist;
    int histsize = 256;
    float range[] = {0, 256};
    const float* histrange = { range };
    split(image1, bgr_planes);
    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histsize, &histrange, true, false);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histsize, &histrange, true, false);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histsize, &histrange, true, false);
    
    int height = 400;
    int width = 500;
    Mat histimage(height, width, CV_8SC3, Scalar(0, 0, 0));
    normalize(b_hist, b_hist, 0, height, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, height, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, height, NORM_MINMAX, -1, Mat());
    
    for(int i = 0; i < histsize; i++)
    {
        line(histimage, Point(i * width / histsize, height - cvRound(b_hist.at<float>(i))), Point((i + 1) * width / histsize, height - cvRound(b_hist.at<float>(i + 1))), Scalar(255, 0, 0), 2, LINE_AA);
        line(histimage, Point(i * width / histsize, height - cvRound(g_hist.at<float>(i))), Point((i + 1) * width / histsize, height - cvRound(g_hist.at<float>(i + 1))), Scalar(0, 255, 0), 2, LINE_AA);
        line(histimage, Point(i * width / histsize, height - cvRound(r_hist.at<float>(i))), Point((i + 1) * width / histsize, height - cvRound(r_hist.at<float>(i + 1))), Scalar(0, 0, 255), 2, LINE_AA);
    }
    imshow("histimage", histimage);
    waitKey(0);
    return 0;
}

#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

string converttostring(double);

int main(int argc, char** argv)
{
    Mat image1, image2;
    int channels[] = {0, 1};
    int histsize[] = {50, 60};
    float range_h[] = {0, 180};
    float range_s[] = {0, 256};
    const float* ranges[] = {range_h, range_s};
    MatND hist1, hist2;
    image1 = imread("/Users/apple/Desktop/00.jpeg");
    image2 = imread("/Users/apple/Desktop/11.jpeg");
    if(!image1.data)
    {
        cout << "could not load image..." << endl;
        return -1;
    }
    
    cvtColor(image1, image1, COLOR_RGB2HSV);
    calcHist(&image1, 1, channels, Mat(), hist1, 2, histsize, ranges);
    normalize(hist1, hist1, 0, 1, NORM_MINMAX, -1, Mat());
    
    cvtColor(image2, image2, COLOR_RGB2HSV);
    calcHist(&image2, 1, channels, Mat(), hist2, 2, histsize, ranges);
    normalize(hist2, hist2, 0, 1, NORM_MINMAX, -1, Mat());
    
    double image1_image2 = compareHist(hist1, hist2, HISTCMP_BHATTACHARYYA);
    cout << "image1 compare with image2 correlation value: " << image1_image2 << endl;
    putText(image2, converttostring(image1_image2), Point(50, 200), FONT_HERSHEY_COMPLEX, 5, Scalar(0, 0, 0), 3 ,LINE_AA);
    imshow("image2", image2);
    
    waitKey(0);
    return 0;
}

string converttostring(double d)
{
    ostringstream os;
    if(os << d)
    {
        return os.str();
    }
    return "invalid conversion";
}

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void histandback(int, void*);
Mat image1, image2, image3, image4;
int bins = 12;

int main(int argc, char** argv)
{
    image1 = imread("/Users/apple/Desktop/00.jpeg");
    if(!image1.data)
    {
        cout << "could not load image..." << endl;
        return -1;
    }
    int from_to[] = {0, 0};
    char input[] = "input image";
    
    cvtColor(image1, image2, COLOR_RGB2HSV);
    image3.create(image2.size(), image3.depth());
    mixChannels(&image2, 1, &image3, 1, from_to, 1);
    
    namedWindow(input, WINDOW_AUTOSIZE);
    imshow(input, image1);
    createTrackbar("Histogram Bins", input, &bins, 180, histandback);
    histandback(0, 0);
    
    waitKey(0);
    return 0;
}

void histandback(int, void*)
{
    float range[] = {0, 180};
    const float *histranges[] = { range };
    Mat histimage;
    calcHist(&image3, 1, 0, Mat(), histimage, 1, &bins, histranges, true, false);
    normalize(histimage, histimage, 0, 255, NORM_MINMAX, -1, Mat());
    Mat backimage;
    calcBackProject(&image3, 1, 0, histimage, backimage, histranges, 1, true);
    imshow("backimage", backimage);
    
    int height = 400;
    int width = 400;
    Mat hist(height, width, CV_8SC3, Scalar(0, 0, 0));
    int bin_w = width / bins;
    for(int i = 0; i < bins; i++)
    {
        rectangle(hist, Point(i * bin_w, height - cvRound(histimage.at<float>(i) * 400 / 255)), Point((i + 1) * bin_w, height), Scalar(0, 0, 255), -1);
    }
    imshow("RectHistImage", hist);
}

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int match_methods = 1;
Mat image1, image2, image4;
char output[] = "image3";
void Match_demo(int, void*);

int main(int argc, char** argv)
{
    image1 = imread("/Users/apple/Desktop/00.jpeg");
    image2 = imread("/Users/apple/Desktop/000.jpeg");
    if(!image1.data || !image2.data)
    {
        cout << "could not load image..." << endl;
        return -1;
    }
    imshow("image1", image1);
    namedWindow(output, WINDOW_AUTOSIZE);
    createTrackbar("Match Type:", output, &match_methods, 5, Match_demo);
    Match_demo(0, 0);
    waitKey(0);
    return 0;
}

void Match_demo(int, void*)
{
    int width = image1.cols - image2.cols + 1;
    int height = image1.rows - image2.rows + 1;
    Mat image3(width, height, CV_32FC3);
    
    matchTemplate(image1, image2, image3, match_methods, Mat());
    normalize(image3, image3, 0, 1, NORM_MINMAX, -1, Mat());
    Point minloc, maxloc, image2loc;
    double min, max;
    image1.copyTo(image4);
    minMaxLoc(image3, &min, &max, &minloc, &maxloc, Mat());
    if(match_methods == TM_SQDIFF || match_methods == TM_SQDIFF_NORMED  )
    {
        image2loc = minloc;
    }
    else
    {
        image2loc = maxloc;
    }
    rectangle(image4, Rect(image2loc.x, image2loc.y, image2.cols, image2.rows), Scalar(0, 0, 255), 2, 8);
    rectangle(image3, Rect(image2loc.x, image2loc.y, image2.cols, image2.rows), Scalar(0, 0, 255), 2, 8);
    
    imshow(output, image3);
    imshow("image4", image4);
}


#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat image1, image2, image3, image4;
int tvalue = 100;
char output[] = "image4";
void contours(int, void*);

int main(int argc, char** argv)
{
    image1 = imread("/Users/apple/Desktop/00.jpeg");
    if(!image1.data)
    {
        cout << "could not load image..." << endl;
        return -1;
    }
    imshow("image1", image1);
    namedWindow(output, WINDOW_AUTOSIZE);
    cvtColor(image1, image2, COLOR_RGB2GRAY);
    createTrackbar("threshold value: ", output, &tvalue, 255, contours);
    contours(0, 0);
    
    waitKey(0);
    return 0;
}
void contours(int, void*)
{
    vector<vector<Point>> points;
    vector<Vec4i> hierarchy;
    Canny(image2, image3, tvalue, tvalue * 2, 3, true);
    findContours(image3, points, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
    image4 = Mat::zeros(image1.size(), CV_8UC3);
    
    RNG rng;
    for(int i = 0; i < points.size(); i++)
    {
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        drawContours(image4, points, i, color, 2, 8, hierarchy, 0, Point(0, 0));
    }
    imshow(output, image4);
}

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat image1, image2, image3, image4;
int thresh = 100;
char blackwhite[] = "image3";
void thresh_demo(int, void*);

int main(int argc, char** argv)
{
    image1 = imread("/Users/apple/Desktop/1.jpeg");
    if(!image1.data)
    {
        cout << "could not load image..." << endl;
        return -1;
    }
    namedWindow(blackwhite, WINDOW_AUTOSIZE);
    imshow("image1", image1);
    cvtColor(image1, image2, COLOR_BGR2GRAY);
    blur(image2, image2, Size(3, 3), Point(-1, -1), BORDER_DEFAULT);
    
    createTrackbar("threshold value:", blackwhite, &thresh, 255, thresh_demo);
    thresh_demo(0, 0);
    
    waitKey(0);
    return 0;
}

void thresh_demo(int, void*)
{
    vector<vector<Point>> points;
    vector<Vec4i> hierarchy;
    image4 = Mat::zeros(image1.size(), CV_8UC3);
    
    threshold(image2, image3, thresh, 255, THRESH_BINARY);
    imshow(blackwhite, image3);
    
    findContours(image3, points, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
    vector<vector<Point>> convexs(points.size());
    
    for(size_t i = 0; i < points.size(); i++)
    {
        convexHull(points[i], convexs[i], false);
    }
    
    for(size_t k = 0; k < points.size(); k++)
    {
        Scalar color = Scalar(255, 255, 255);
        drawContours(image4, points, k, color, 2, LINE_8, hierarchy, 0, Point(0, 0));
        drawContours(image4, convexs, k, color, 2, LINE_8, hierarchy, 0, Point(0, 0));
    }
    
    imshow("image4", image4);
}

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat image1, image2, image3, image4;
char blackwhite[] = "image3";
int thresh = 100;
void draw_demo(int, void*);

int main(int argc, char** argv)
{
    image1 = imread("/Users/apple/Desktop/1.jpeg");
    if(!image1.data)
    {
        cout << "could not load image..." << endl;
        return -1;
    }
    namedWindow(blackwhite, WINDOW_AUTOSIZE);
    cvtColor(image1, image2, COLOR_RGB2GRAY);
    blur(image2, image2, Size(3, 3), Point(-1, -1), BORDER_DEFAULT);
    
    createTrackbar("Threshold value:", blackwhite, &thresh, 255, draw_demo);
    draw_demo(0, 0);
    
    waitKey(0);
    return 0;
}

void draw_demo(int, void*)
{
    vector<vector<Point>> points;
    vector<Vec4i> hierarchy;
    
    threshold(image2, image3, thresh, 255, THRESH_BINARY);
    imshow(blackwhite, image3);
    
    findContours(image3, points, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(-1, -1));
    
    vector<vector<Point>> contour_poly(points.size());
    vector<Rect> contour_rect(points.size());
    vector<Point2f> center(points.size());
    vector<float> radius(points.size());
    vector<RotatedRect> Rects(points.size());
    vector<RotatedRect> Ellipses(points.size());
    
    for(size_t i = 0; i < points.size(); i++)
    {
        approxPolyDP(points[i], contour_poly[i], 3, true);
        contour_rect[i] = boundingRect(contour_poly[i]);
        minEnclosingCircle(contour_poly[i], center[i], radius[i]);
        if(contour_poly[i].size() > 5)
        {
            Ellipses[i] = fitEllipse(contour_poly[i]);
            Rects[i] =  minAreaRect(contour_poly[i]);
        }
    }
    image1.copyTo(image4);
    Point2f pts[4];
    for(size_t k = 0; k < points.size(); k++)
    {
        Scalar color = Scalar(255, 255, 255);
        //rectangle(image4, contour_rect[k], color, 2, 8);
        //circle(image4, center[k], radius[k], color, 2, 8);
        if(contour_poly[k].size() > 5)
        {
            ellipse(image4, Ellipses[k], color);
            Rects[k].points(pts);
            for(int t = 0; t < 4; t++)
            {
                line(image4, pts[t], pts[(t+1) % 4], color);
            }
        }
    }
    imshow("image4", image4);
}

//毕业设计
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int value = 80;
char blackandwhite[] = "BIN_PTCR";
Mat image1, image2, image3, image4;
void thres_demo(int, void*);

int main(int argc, char** argv)
{
    image1 = imread("/Users/apple/Desktop/3.jpeg");
    if(!image1.data)
    {
        cout << "could not load image..." << endl;
        return -1;
    }
    imshow("PTCR", image1);
    cvtColor(image1, image2, COLOR_RGB2GRAY);
    GaussianBlur(image2, image2, Size(5, 5), 0, 0);
    namedWindow(blackandwhite, WINDOW_AUTOSIZE);
    createTrackbar("threshold value:", blackandwhite, &value, 255, thres_demo);
    thres_demo(0, 0);
    
    waitKey(0);
    return 0;
}

void thres_demo(int, void*)
{
    threshold(image2, image3, value, 255, THRESH_BINARY);
    //Canny(image2, image3, value, value * 2);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(9, 9));
    erode(image3, image3, kernel);
    dilate(image3, image3, kernel);
    //bitwise_not(image3, image3);
    imshow(blackandwhite, image3);
    
    vector<vector<Point>> points;
    vector<Vec4i> hierarchy;
    findContours(image3, points, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    
    vector<vector<Point>> contour_poly(points.size());
    vector<Rect> contour_rect(points.size());
    int x0=0, y0=0, w0=0, h0=0;
    for(size_t i = 0; i < points.size(); i++)
    {
        approxPolyDP(points[i], contour_poly[i], 3, true);
        contour_rect[i] =  boundingRect(contour_poly[i]);
        if(contour_rect[i].height > 900 || contour_rect[i].width > 900)
        {
            x0 = contour_rect[i].x;
            y0 = contour_rect[i].y;
            w0 = contour_rect[i].width;
            h0 = contour_rect[i].height;
        }
    }
    Mat ROI = image3(Rect(x0, y0, w0, h0));
    
    imshow("ROIimage", ROI);
}*/

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;


int main(int argc, char** argv)
{
    Mat image1 = imread("/Users/apple/Desktop/00.jpeg");
    if(!image1.data)
    {
        cout << "could not load image..." << endl;
        return -1;
    }
    imshow("image1", image1);
    
    
    
    waitKey(0);
    return 0;
}
