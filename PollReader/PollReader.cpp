/**
* @function goodFeaturesToTrack_Demo.cpp
* @brief Demo code for detecting corners using Shi-Tomasi method
* @author OpenCV team
*/

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d.hpp"
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
void operation(int, void*);
void show(Mat& img);
void score();
void loadTemplate();
void report();

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
	loadTemplate();

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
	std::vector<cv::Vec4f> lines;
	cv::HoughLinesP(bw, lines, 1, CV_PI / 180, iThreshold, src.cols / 2, 30);
	double angle = 0.0;
	unsigned nb_lines = lines.size();
	for (unsigned i = 0; i < nb_lines; ++i)
	{
		cv::line(src, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(255, 0, 0));
		angle += atan2((double)lines[i][3] - lines[i][1], (double)lines[i][2] - lines[i][0]);
	}
	angle /= nb_lines; // mean angle, in radians.
	show(src);

	/// Deskew
	Mat rM = cv::getRotationMatrix2D(cv::Point2f(0, 0), angle/CV_PI*180.0, 1.0);
	Mat rotated;
	cv::warpAffine(bw, rotated, rM, cv::Size(src.cols, src.rows));
	show(rotated);

	/// Crop
	cv::Rect rect(area[0], area[1], area[2], area[3]);
	cropped = rotated(rect);
	show(cropped);

	/// Erode
	int erode_size = 3;
	Mat element = getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE,
		Size(2 * erode_size + 1, 2 * erode_size + 1),
		Point(erode_size, erode_size));
	cv::erode(cropped, eroded, element);
	show(eroded);

	/// Blobs Detect
	cv::SimpleBlobDetector::Params params;
	params.minDistBetweenBlobs = 50.0f;
	params.filterByInertia = false;
	params.filterByConvexity = false;
	params.filterByColor = false;
	params.filterByCircularity = false;
	params.filterByArea = false;
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
			cv::line(result, cv::Point(x, 0), cv::Point(x, area[3]), cv::Scalar(255, 255, 255, 255), 3);
		}
		for (size_t i = 0; i < questions.size(); i++)
		{
			float y = questions[i];
			cv::line(result, cv::Point(0, y), cv::Point(area[2], y), cv::Scalar(255, 255, 255, 255), 3);
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

void show(Mat& img)
{
	if (interactive)
	{
		imshow(main_window, img);
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
	scores.resize(questions.size());

	std::sort(keypoints.begin(), keypoints.end(), sortKeypoint);
	for (size_t i = 0; i < keypoints.size(); i++)
	{
		KeyPoint& kp = keypoints[i];
		nQ = getQuestionNumber(kp.pt.y);

		if (nQ >= 0)
		{
			nC = getChoiceNumber(kp.pt.x);
			scores[nQ] = nC;
		}
	}

}