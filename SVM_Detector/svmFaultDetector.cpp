#include "svmDetector.h"

using namespace cv;
using namespace std;

int main(int argc, const char** argv)
{
	// Creating a keymap for all the arguments that can passed to that programm
	const String keyMap = "{help h usage ?  |   | show this message}"
						  "{svm  |   | path for the trained svm}"
						  "{save  |   | name for the detect result}"
						  "{vol  volume |   | path for the volume in which find fault}";

	// Reading the calling arguments
	CommandLineParser parser(argc, argv, keyMap);
	parser.about("Fault_Training");

	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	String svmPath = parser.get<String>("svm");
	String saveFile = parser.get<String>("save");
	String volumePath = parser.get<String>("vol");

	if (svmPath == "")
	{
		printf("There is no svm Path\n");
		return -1;
	}
	if (volumePath == "")
	{
		printf("There is no input files to detect fault\n");
		return -1;
	}
	if (saveFile == "")
	{
		printf("There is no file to save detect result\n");
		return -1;
	}

	Ptr<ml::SVM> svm = ml::SVM::create();

	svm = svm->load<ml::SVM>(svmPath);
	if (!svm->isTrained())
	{
		printf("The SVM isn't trained through this path: %s\n", svmPath);
		return -1;
	}


	bool detect = detectFaultSVM(svm, &volumePath, &saveFile);
	if (!detect)
		return -1;
	return 0;
}


bool detectFaultSVM(Ptr<ml::SVM> svm, String* volumePath, String* saveFile)
{
	printf("Initialize\n");

	// Finding all images in both pathes
	std::vector<String> dataFileNames;
	glob(*volumePath, dataFileNames);

	// Testing if there are images in the pathes
	if (dataFileNames.size() <= 0)
	{
		printf("There are no volume data %s\n", *volumePath);
		return false;
	}

	std:vector<float> detectLabel(dataFileNames.size());
	Mat detectData = Mat_<float>(SEISMIC_SUBVOLUME_SIZE, dataFileNames.size());
	int detectCount = 0;


	clock_t beginTime = clock();

	std::cout << "Read attributes (" << (clock() - beginTime) / (float)CLOCKS_PER_SEC << ") ...";

        std::vector<float> tmpData;
	for (std::vector<String>::iterator fileName = dataFileNames.begin(); fileName != dataFileNames.end(); ++fileName)
	{
            FILE *fdata = fopen((*fileName).c_str(),"r");
            fread(&tmpData[0],SEISMIC_SUBVOLUME_SIZE*4, 1, fdata);
            fclose(fdata);
            float rawPrediction = svm->predict(tmpData, noArray(), ml::StatModel::RAW_OUTPUT);
            if (rawPrediction < 0)
                detectLabel[detectCount] = rawPrediction; 
	    detectCount++;
	}
	std::cout << " Finished (" << (clock() - beginTime) / (float)CLOCKS_PER_SEC << ")" << std::endl;

        FILE *flabel = fopen((*saveFile).c_str(),"w");
        int labelCount=0;
        while(labelCount<detectCount)
        {
            float v = detectLabel[labelCount];
            fwrite((void*)(&v), sizeof(v), 1, flabel);
            labelCount++;
        }
        fclose(flabel);

	return true;
}
