#include "svmDetector.h"

using namespace cv;
using namespace std;

int main(int argc, const char** argv)
{
	// Creating a keymap for all the arguments that can passed to that programm
	const String keyMap = "{help h usage ?  |   | show this message}"
						  "{svm  |   | path for the trained svm}"
						  "{data  |   | name for the detect result}"
						  "{save  volume |   | path for the volume in which find fault}";

	// Reading the calling arguments
	CommandLineParser parser(argc, argv, keyMap);
	parser.about("Fault_Training");

	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	String svmFile = parser.get<String>("svm");
	String dataPath = parser.get<String>("data");
	String saveFile= parser.get<String>("save");

	if (svmFile == "")
	{
		printf("There is no svm file\n");
		return -1;
	}
	if (dataPath == "")
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

	svm = svm->load<ml::SVM>(svmFile);
	if (!svm->isTrained())
	{
		printf("The SVM isn't trained through this path: %s\n", svmFile);
		return -1;
	}


	bool detect = detectFaultSVM2(svm, &dataPath, &saveFile);
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

long GetFileSize(std::string filename)
{
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

bool detectFaultSVM2(Ptr<ml::SVM> svm, String* volumePath, String* saveFile)
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
        
        string firstName = *(dataFileNames.begin());
        long sampleSize = GetFileSize(firstName)/4;

	std:vector<float> detectLabel(sampleSize);
	Mat detectData = Mat_<float>(sampleSize, dataFileNames.size());
	int detectCount = 0;

        printf("%s:%d\n",__FUNCTION__,__LINE__);
        int fileIdx = 0;
        for (std::vector<String>::iterator fileName = dataFileNames.begin(); fileName != dataFileNames.end(); ++fileName)
        {
            //std:vector<float> sampleData(sampleSize);
            FILE *fsample = fopen((*fileName).c_str(),"r");
            int sampleCount=0;
            printf("%s:%d,read from:%s\n",__FUNCTION__,__LINE__,(*fileName).c_str());
            while( sampleCount<sampleSize && (!feof(fsample)) )
            {
                float v;
                unsigned char temp[4];
                fread((void*)(&temp[0]), sizeof(temp[0]), 1, fsample);
                fread((void*)(&temp[1]), sizeof(temp[1]), 1, fsample);
                fread((void*)(&temp[2]), sizeof(temp[2]), 1, fsample);
                fread((void*)(&temp[3]), sizeof(temp[3]), 1, fsample);
                //int itemp = (temp[0]<<0) | (temp[1]<<8) | (temp[2]<<16) | (temp[3]<<24);
                unsigned int itemp =  (temp[3]<<0) | (temp[2]<<8) | (temp[1]<<16) | (temp[0]<<24);
                v = *((float*)&itemp);
                //sampleData[sampleCount] = v;
                //printf("%s:%d, file:%d,sample:%d:%f vs: %f\n",__FUNCTION__,__LINE__,fileIdx,sampleCount,v,tempLabel.at<float>(sampleCount,0));
                detectData.at<float>(sampleCount,fileIdx) = v;
                sampleCount++;
            }
            printf("%s:%d,read finish:%s,%d\n",__FUNCTION__,__LINE__,(*fileName).c_str(),sampleCount);
            fclose(fsample);
            fileIdx++;
        }

        printf("%s:%d\n",__FUNCTION__,__LINE__);

	clock_t beginTime = clock();

	std::cout << "Read attributes (" << (clock() - beginTime) / (float)CLOCKS_PER_SEC << ") ...";

        std::vector<float> tmpData(dataFileNames.size());
	for (int i=0;i<sampleSize;i++)
	{
            printf("%d:%d",__LINE__,i);
            for(int j=0;j<dataFileNames.size();j++)
            {
               tmpData[j] = detectData.at<float>(i,j); 
               printf(",%f",tmpData[j]);
            }
            printf("\n");
            float rawPrediction = svm->predict(tmpData, noArray(), ml::StatModel::RAW_OUTPUT);
            //printf("Predict:%d, %f => %f \n",__LINE__,tmpData[0],rawPrediction);
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


