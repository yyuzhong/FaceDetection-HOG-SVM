// -svm=D:\Dokumente\Workspaces\C++_VS\CV2_Ex3\SVM_Train\SVM_MARC.yaml -img=D:\Dokumente\Workspaces\C++_VS\CV2_Ex3\data\testImages\detection\t1.jpg
#include "svmDetector.h"

using namespace cv;
using namespace std;
typedef int64_t INT64;
#include <sys/time.h>
#include <ctime>

int main(int argc, const char** argv)
{
#pragma region Argument parsing

	// Creating a keymap for all the arguments that can passed to that programm
	const String keyMap = "{help h usage ?  |   | show this message}"
						  "{svm |   | path to the trained svm}"
						  "{img image  |   | path for the image in wich it find the faces}"
						  "{cam |  | for the activation of the webcam mode }";

	// Reading the calling arguments
	CommandLineParser parser(argc, argv, keyMap);
	parser.about("FaceDetection-HOG-SVM");

	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	String svmPath = parser.get<String>("svm");
	String imagePath = parser.get<String>("img");
	bool camActivation = parser.has("cam");

#pragma endregion

#pragma region Initialization

	Ptr<ml::SVM> svm = ml::SVM::create();
	svm = svm->load<ml::SVM>(svmPath);

	if (!svm->isTrained())
	{
		printf("The SVM isn't trained through this path: %s\n", svmPath);
		return -1;
	}

	VideoCapture capture; 
	Mat inputImage;
	if (camActivation)
	{
		capture = VideoCapture(0);
		if (!capture.isOpened())
		{
			printf("There is no cam to get the stream\n");
			return -1;
		}
	}
	else {
		if (imagePath == "")
		{
			printf("There is no positivePath\n");
			return -1;
		}	

		inputImage = imread(imagePath);
		if (inputImage.empty())
		{
			printf("This is no image: %s\n", imagePath);
			return -1;
		}
	}

#pragma endregion

#pragma region Function Decesion

	if (camActivation)
	{
		int result = webcamDetection(&capture, svm);
		if (result != 0)
			return -1;
		return 0;
	}
	else
	{
		int result = imageDetection(&inputImage, svm);
		if (result != 0)
			return -1;
		return 0;
	}

#pragma endregion
}

bool saveMatToFile(Mat inputMat, string outputFile)
{
    vector<int> p;
    p.push_back(CV_IMWRITE_JPEG_QUALITY);
    p.push_back(100);
    vector<unsigned char> buf;

    if(inputMat.data == NULL)
    {
        cout << "Nothing is save for NULL data" << endl;
	return false;
    }
    
    if((inputMat.cols > 1920)||(inputMat.rows > 1080))
    {
        float xfact = (float)1920.0/inputMat.cols;
        float yfact = (float)1280.0/inputMat.rows;
        resize(inputMat,inputMat,Size(),min(xfact,yfact),min(xfact,yfact));
    }
    
    cv::imencode(".jpg", inputMat, buf, p);

    const char *cfile = outputFile.c_str();
    FILE* localFileHandle = fopen(cfile, "wb");
    fwrite((void*)&buf[0], (unsigned int)buf.size(),1,localFileHandle);
    fclose(localFileHandle);
    
    return true;
}



int imageDetection(Mat* inputImage, Ptr<ml::SVM> svm)
{
#pragma region Detect in image

	Mat outputImage = faceDetection(inputImage, svm);
	if (outputImage.empty())
		return -1;

	//namedWindow("Face Detection");
	//imshow("Face Detection", outputImage);

	//waitKey();
        saveMatToFile(outputImage,"/var/www/html/imgproc/testout.jpg");	

	return 0;

#pragma endregion
}

int webcamDetection(VideoCapture* capture, Ptr<ml::SVM> svm)
{
#pragma region Detect in Webcam stream

	namedWindow("Webcame Face Detection");
	Mat frameImage;
	while (true)
	{
		*capture >> frameImage;
		resize(frameImage, frameImage, Size(frameImage.cols * 1/2, frameImage.rows * 1/2));
		imshow("Webcame Face Detection", faceDetection(&frameImage, svm));

		if (waitKey(50) >= 0) 			
			return 0;
	}

#pragma endregion
}

Mat faceDetection(Mat* inputImage, Ptr<ml::SVM> svm)
{
#pragma region Initialization

	Mat greyImage;
	cvtColor(*inputImage, greyImage, CV_BGR2GRAY);

	// Vector that saves the Point in whicht the match was and the preditcion value and the scale factor
	std::vector<std::pair<Point, Vec2f> > positivePatches;

	HOGDescriptor hogD;
	hogD.winSize = Size(WINDOW_SIZE, WINDOW_SIZE);

	clock_t beginTime = clock();

#pragma endregion 

#pragma region Face Detection

	Mat scaledImage = greyImage;
	float scaleFactor = 1;
	std::cout << "Begin the face detection (" << (clock() - beginTime) / (float)CLOCKS_PER_SEC << ") ...";

	while (scaledImage.rows >= WINDOW_SIZE && scaledImage.cols >= WINDOW_SIZE)
	{
#pragma omp parallel for
		for (int cY = 0; cY < (scaledImage.rows - WINDOW_SIZE); cY += PATCH_PIXEL_MOVEMENT)
		{
			std::vector<float> descriptorsValues;
			std::vector<Point> locations;
#pragma omp parallel for
			for (int cX = 0; cX < (scaledImage.cols - WINDOW_SIZE); cX += PATCH_PIXEL_MOVEMENT)
			{
				// Take the patch from the image
				Mat imagePatch = scaledImage(Range(cY, cY + WINDOW_SIZE), Range(cX, cX + WINDOW_SIZE));
				// Calculating the HOG
				hogD.compute(imagePatch, descriptorsValues, Size(0, 0), Size(0, 0), locations);
				// Predict with the SVM
				float rawPrediction = svm->predict(descriptorsValues, noArray(), ml::StatModel::RAW_OUTPUT);
#pragma omp critical
				{
					if (rawPrediction < 0)
						positivePatches.push_back(std::pair<Point, Vec2f>(Point(cX, cY), Vec2f(rawPrediction, scaleFactor)));
				}
			}
		}
		// Donwscale the image
		resize(scaledImage, scaledImage, Size(scaledImage.cols * DOWNSCALE_FACTOR, scaledImage.rows * DOWNSCALE_FACTOR));
		// Save the new scalfactor (zoomfactor)
		scaleFactor /= DOWNSCALE_FACTOR;
	}
	std::cout << " Finished (" << (clock() - beginTime) / (float)CLOCKS_PER_SEC << ")" << std::endl;

#pragma endregion

#pragma region Draw Boxes in the image

	// Sort the vector
	std::sort(positivePatches.begin(), positivePatches.end(), sortPreditcionVector);

	Mat outputImage = *inputImage;
	std::cout << "Begin the drawing (" << (clock() - beginTime) / (float)CLOCKS_PER_SEC << ") ...";
	for (std::vector<std::pair<Point, Vec2f> >::iterator patches = positivePatches.begin(); patches != positivePatches.end(); ++patches)
	{
		// Get the upper-left und lower-right point for the rect 
		Point rectPointUL = patches->first * patches->second[1];
		Point rectPointLR = rectPointUL + (Point(64, 64) * patches->second[1]);
		// Draw the rectangle in the image
		if (patches == positivePatches.begin())
			rectangle(outputImage, rectPointUL, rectPointLR, Scalar(150, 100, 200), 3);
		else
			rectangle(outputImage, rectPointUL, rectPointLR, Scalar(0, 0, 255));
	}
	std::cout << " Finished (" << (clock() - beginTime) / (float)CLOCKS_PER_SEC << ")" << std::endl;

#pragma endregion

	return outputImage;

}

bool sortPreditcionVector(std::pair<Point, Vec2f> left, std::pair<Point, Vec2f> right)
{
	return left.second[0] < right.second[0];
}
