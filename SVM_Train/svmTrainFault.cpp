#include "svmTraining.h"

using namespace cv;
using namespace std;

int main(int argc, const char** argv)
{
	// Creating a keymap for all the arguments that can passed to that programm
	const String keyMap = "{help h usage ?  |   | show this message}"
						  "{data d data  |   | path for the training data}"
						  "{label l label  |   | file for the training label}";

	// Reading the calling arguments
	CommandLineParser parser(argc, argv, keyMap);
	parser.about("Fault_Training");

	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	String dataPath = parser.get<String>("data");
	String labelFile = parser.get<String>("label");

	if (dataPath == "")
	{
		printf("There is no dataPath\n");
		return -1;
	}
	if (labelFile == "")
	{
		printf("There is no labelFile\n");
		return -1;
	}

	bool train = trainFaultSVM(&dataPath, &labelFile);
	if (!train)
		return -1;
	return 0;
}


bool trainFaultSVM(String* dataTrainPath, String* labelTrainFile)
{
	printf("Initialize\n");
	// Finding all images in both pathes
	std::vector<String> dataFileNames;
	glob(*dataTrainPath, dataFileNames);

	// Testing if there are images in the pathes
	if (dataFileNames.size() <= 0)
	{
		printf("There are no trainning data %s\n", *dataTrainPath);
		return false;
	}
	if (!labelTrainFile || (*labelTrainFile).length()<=0)
	{
		printf("There are no label file\n");
		return false;
	}

	Mat trainingLabel = Mat_<int>(1, dataFileNames.size());
	Mat trainingData = Mat_<float>(SEISMIC_SUBVOLUME_SIZE, dataFileNames.size());
	int trainingCount = 0;


	clock_t beginTime = clock();

        FILE *flabel = fopen((*labelTrainFile).c_str(),"r");
        int labelCount=0;
        while(!feof(flabel))
        {
            float v;
            fread((void*)(&v), sizeof(v), 1, flabel);
            trainingLabel.at<int>(0, labelCount) = v>0.5?1:0;
            labelCount++;
        }
        fclose(flabel);
	std::cout << "Read attributes (" << (clock() - beginTime) / (float)CLOCKS_PER_SEC << ") ...";
        //float tmpData[SEISMIC_SUBVOLUME_SIZE];
        std::vector<float> tmpData;
	for (std::vector<String>::iterator fileName = dataFileNames.begin(); fileName != dataFileNames.end(); ++fileName)
	{
            FILE *fdata = fopen((*fileName).c_str(),"r");
            fread(&tmpData[0],SEISMIC_SUBVOLUME_SIZE*4, 1, fdata);
	    Mat descriptorsVector = Mat_<float>(tmpData, true);
	    descriptorsVector.col(0).copyTo(trainingData.col(trainingCount));
	    trainingCount++;
	}
	std::cout << " Finished (" << (clock() - beginTime) / (float)CLOCKS_PER_SEC << ")" << std::endl;

	// Set up SVM's parameters
	Ptr<ml::SVM> svm = ml::SVM::create();
	svm->setType(ml::SVM::C_SVC);
	svm->setKernel(ml::SVM::LINEAR);
	svm->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, SVM_ITERATIONS, 1e-6));
	// Create the Trainingdata
	Ptr<ml::TrainData> tData = ml::TrainData::create(trainingData, ml::SampleTypes::COL_SAMPLE, trainingLabel);

	std::cout << "Start SVM training (" << (clock() - beginTime) / (float)CLOCKS_PER_SEC << ") ...";
	svm->train(tData);
	std::cout << " Finished (" << (clock() - beginTime) / (float)CLOCKS_PER_SEC << ")" << std::endl;

	svm->save(SVM_OUTPUT_NAME);

	return true;
}
