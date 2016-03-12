/**
* @function goodFeaturesToTrack_Demo.cpp
* @brief Demo code for detecting corners using Shi-Tomasi method
* @author OpenCV team
*/

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/photo/photo.hpp"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

/// Global variables
bool interactive = true;
char* sourcePath;
char* templatePath;
char* resultPath;
Mat src, gray, bw, rotated, cropped, eroded, result;
int iThreshold = 100;
const char* main_window = "Image";
std::vector<cv::KeyPoint> keypoints;
std::vector<float> area;
std::vector<int> choices;
std::vector<int> questions;
std::vector<int> scores;

/// Function header
void show(Mat& img);
void score();
void loadTemplate();
void report();

bool sortLine(cv::Vec4f a, cv::Vec4f b)
{
	float lengthSqrA = (a[2] - a[0])*(a[2] - a[0]) + (a[3] - a[1])*(a[3] - a[1]);
	float lengthSqrB = (b[2] - b[0])*(b[2] - b[0]) + (b[3] - b[1])*(b[3] - b[1]);
	return lengthSqrA > lengthSqrB;
}

/**
* @function main
*/
int main(int argc, char** argv)
{
	if (argc < 4) 
	{
		cout<<"Usage: PollReader.exe SourcePath TemplatePath ResultPath [isInteractive]"<<endl;
		return 0;
	}

	sourcePath = argv[1];
	templatePath = argv[2];
	resultPath = argv[3];
	interactive = argc == 5 ? true: false;

	/// Load Template
	//loadTemplate();

	/// Create Window
	if (interactive)
	{
		namedWindow(main_window, WINDOW_NORMAL);
	}

	/// Load source image and convert it to gray
	src = imread(sourcePath, 1);
	cvtColor(src, gray, COLOR_BGR2GRAY);

	/// Detect Skew angle
	cv::threshold(gray, bw, iThreshold, 255.0, THRESH_BINARY_INV);
	std::vector<cv::Vec4f> lines, longLines;
	cv::HoughLinesP(bw, lines, 1, CV_PI / 180, iThreshold, src.cols / 2, 30);
	double angle = 0.0;
	unsigned nb_lines = lines.size();
	std::sort(lines.begin(), lines.end(), sortLine);
	
	for (unsigned i = 0; i < nb_lines; ++i)
	{
		cv::Vec4f line = lines[i];
		if (i < 40 && line[1] < 500)
		{
			longLines.push_back(line);
		}
		cv::Scalar color = cv::Scalar(255, 0, 0);
		cv::line(src, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), color);
		angle += atan2((double)line[3] - line[1], (double)line[2] - line[0]);
	}
	angle /= nb_lines; // mean angle, in radians.
	std::sort(longLines.begin(), longLines.end(), [](cv::Vec4f a, cv::Vec4f b)->bool {return a[1] < b[1]; });
	cv::Vec4f theLine = longLines[longLines.size() / 2];
	cv::Scalar color = cv::Scalar(0, 0, 255);
	cv::line(src, cv::Point(theLine[0], theLine[1]), cv::Point(theLine[2], theLine[3]), color, 5);
	show(src);

	/// Deskew
	Mat rM = cv::getRotationMatrix2D(cv::Point2f(0, 0), angle/CV_PI*180.0, 1.0);
	Mat rotated;
	cv::warpAffine(bw, rotated, rM, cv::Size(src.cols, src.rows));
	cv::warpAffine(gray, gray, rM, cv::Size(src.cols, src.rows));
	show(rotated);

	/// Calculate Cropping Area
	vector<Vec3f> circles, choiceCircles;
	GaussianBlur(gray, gray, Size(9, 9), 2, 2);
	HoughCircles(gray, circles, CV_HOUGH_GRADIENT, 1, 10, 100, 40, 10, 20);
	for (size_t i = 0; i < circles.size(); i++)
	{
		if (circles[i][1] > theLine[1] + 150)
		{
			choiceCircles.push_back(circles[i]);
		}
	}
	std::sort(choiceCircles.begin(), choiceCircles.end(), [](cv::Vec3f a, cv::Vec3f b)->int {return a[0] < b[0]; });
	float minX = choiceCircles[0][0] - 30;
	float maxX = choiceCircles[choiceCircles.size() - 1][0] + 20;
	float currentX = 0.0f;
	float prevX = minX;
	choices.clear();
	for (size_t i = 0; i < choiceCircles.size(); i++)
	{
		cv::Vec3f c = choiceCircles[i];
		float x = c[0];
		if (x > currentX + 10)
		{
			prevX = x + x - prevX;
			choices.push_back(prevX - minX);
		}
		currentX = x;
	}

	std::sort(choiceCircles.begin(), choiceCircles.end(), [](cv::Vec3f a, cv::Vec3f b)->int {return a[1] < b[1]; });
	float minY = choiceCircles[0][1] - 20;
	float maxY = choiceCircles[choiceCircles.size() - 1][1] + 20;
	float currentY = 0.0f;
	float prevY = minY;
	questions.clear();
	for (size_t i = 0; i < choiceCircles.size(); i++)
	{
		cv::Vec3f c = choiceCircles[i];
		float y = c[1];
		if (y > currentY + 10)
		{
			prevY = y + y - prevY;
			questions.push_back(prevY - minY);
		}
		currentY = y;
	}

	/// Crop
	cv::Rect rect(minX, minY, maxX - minX, maxY - minY);
	cropped = rotated(rect);
	show(cropped);

	/// Erode
	int dilate_size = 1;
	int erode_size = 5;
	Mat element = getStructuringElement(cv::MorphShapes::MORPH_RECT,
		Size(2 * dilate_size + 1, 2 * dilate_size + 1),
		Point(-1, -1));
	cv::dilate(cropped, eroded, element);
	element = getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE,
		Size(2 * erode_size + 1, 2 * erode_size + 1),
		Point(-1, -1));
	show(eroded);
	cv::erode(eroded, eroded, element);
	show(eroded);

	/// Blobs Detect
	cv::SimpleBlobDetector::Params params;
	params.minDistBetweenBlobs = 50.0f;
	params.filterByInertia = false;
	params.filterByConvexity = false;
	params.filterByCircularity = false;
	params.filterByColor = false;
	params.filterByArea = true;
	params.minConvexity = 0.1;
	params.minArea = 50;
	params.maxArea = 800;
	//params.minCircularity = 0.5;
	//params.minThreshold = 120.0;
	cv::Ptr<cv::SimpleBlobDetector> blobDetector = cv::SimpleBlobDetector::create(params);
	blobDetector->detect(eroded, keypoints);

	/// Draw Results
	if (interactive)
	{
		result = cropped.clone();
		for (size_t i = 0; i < keypoints.size(); i++)
		{
			cv::circle(result, keypoints[i].pt, 30, cv::Scalar(255, 255, 255, 255), 3);
		}
		for (size_t i = 0; i < choices.size(); i++)
		{
			float x = choices[i];
			cv::line(result, cv::Point(x, 0), cv::Point(x, maxY - minY), cv::Scalar(255, 255, 255, 255), 3);
		}
		for (size_t i = 0; i < questions.size(); i++)
		{
			float y = questions[i];
			cv::line(result, cv::Point(0, y), cv::Point(maxX - minX, y), cv::Scalar(255, 255, 255, 255), 3);
		}
		show(result);
	}

	/// Score
	score();

	/// Print Report
	report();
	for (int i = 0; i < questions.size(); i++)
	{
		cout<<scores[i]<<" ";
	}
	cout<<endl;

	return(0);
}

void report()
{
	ofstream resultStream;
	resultStream.open(resultPath, std::ofstream::app);

	for (int i = 0; i < questions.size(); i++)
	{
		resultStream<<scores[i]<<",";
	}
	resultStream<<endl;
}

void loadTemplate()
{
	ifstream templateStream;
	templateStream.open(templatePath, std::ifstream::in);
	float f = 0;
	int v = 0;
	
	for(int i=0; i<4; i++)
	{
		templateStream>>f;
		area.push_back(f);
	}

	int numChoices = 0;
	templateStream>>numChoices;
	for(int i=0; i<numChoices; i++)
	{
		templateStream>>v;
		choices.push_back(v);
	}

	int numQuestions = 0;
	templateStream>>numQuestions;
	for(int i=0; i<numQuestions; i++)
	{
		templateStream>>v;
		questions.push_back(v);
	}
}

void onMouse(int event, int x, int y, int flags, void* param)
{
	Mat* img = (Mat*)param;
    char text[100];
    Mat img2, img3;

    img2 = img->clone();

    if (event == CV_EVENT_LBUTTONDOWN)
    {
        Vec3b p = img2.at<Vec3b>(y,x);
        sprintf(text, "R=%d, G=%d, B=%d", p[2], p[1], p[0]);
    }
    else if (event == CV_EVENT_RBUTTONDOWN)
    {
        cvtColor(*img, img3, CV_BGR2HSV);
        Vec3b p = img3.at<Vec3b>(y,x);
        sprintf(text, "H=%d, S=%d, V=%d", p[0], p[1], p[2]);
    }
    else
        sprintf(text, "x=%d, y=%d", x, y);

	cv::rectangle(img2, Rect(5, 20, 400, 100), CV_RGB(0,0,0), CV_FILLED);
    cv::putText(img2, text, Point(5,50), FONT_HERSHEY_PLAIN, 2.0, CV_RGB(255,255,255), 2.0);
    imshow(main_window, img2);
}

void show(Mat& img)
{
	if (interactive)
	{
		imshow(main_window, img);
		setMouseCallback(main_window, onMouse, &img);
		waitKey(0);
	}
}

int sortKeypoint(KeyPoint& p1, KeyPoint& p2)
{
	return p1.pt.y < p2.pt.y;
}

int getQuestionNumber(float y)
{
	for (int i = 0; i < questions.size(); i++)
	{
		if (y < questions[i])
			return i;
	}

	return -1;
}

int getChoiceNumber(float x)
{
	for (int i = 0; i < choices.size(); i++)
	{
		if (x < choices[i])
			return i;
	}

	return -1;
};

void score()
{
	int nQ, nC;
	scores.resize(questions.size(), -1);

	std::sort(keypoints.begin(), keypoints.end(), sortKeypoint);
	for (size_t i = 0; i < keypoints.size(); i++)
	{
		KeyPoint& kp = keypoints[i];
		nQ = getQuestionNumber(kp.pt.y);

		if (nQ >= 0)
		{
			nC = getChoiceNumber(kp.pt.x);
			scores[nQ] = (scores[nQ] > 0) ? -1 : nC;
		}
	}

}