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

	bool train = trainFaultSVM3(&dataPath, &labelFile);
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

        printf("%s:%d\n",__FUNCTION__,__LINE__);

	Mat trainingLabel = Mat_<int>(1, dataFileNames.size());
	Mat trainingData = Mat_<float>(SEISMIC_SUBVOLUME_SIZE, dataFileNames.size());
	int trainingCount = 0;

        printf("%s:%d\n",__FUNCTION__,__LINE__);

	clock_t beginTime = clock();

        FILE *flabel = fopen((*labelTrainFile).c_str(),"r");
        int labelCount=0;
        while( labelCount<dataFileNames.size() && (!feof(flabel)) )
        {
            float v;
            fread((void*)(&v), sizeof(v), 1, flabel);
            printf("%s:%d, %d:%f\n",__FUNCTION__,__LINE__,labelCount,v);
            trainingLabel.at<int>(0, labelCount) = v>0.5?1:0;
            /*Only for Debug!!!*/ if(labelCount%10==0) trainingLabel.at<int>(0, labelCount) = 1;
            labelCount++;
        }
        fclose(flabel);
        printf("%s:%d\n",__FUNCTION__,__LINE__);

	//std::cout << "Read attributes (" << (clock() - beginTime) / (float)CLOCKS_PER_SEC << ") ...";
        //float tmpData[SEISMIC_SUBVOLUME_SIZE];
        std::vector<float> tmpData(SEISMIC_SUBVOLUME_SIZE);
	for (std::vector<String>::iterator fileName = dataFileNames.begin(); fileName != dataFileNames.end(); ++fileName)
	{
            FILE *fdata = fopen((*fileName).c_str(),"r");
            //printf("%s:%d,%s\n",__FUNCTION__,__LINE__,(*fileName).c_str());
            fread(&tmpData[0],SEISMIC_SUBVOLUME_SIZE*4, 1, fdata);
            //printf("%s:%d,%d\n",__FUNCTION__,__LINE__,trainingCount);
	    Mat descriptorsVector = Mat_<float>(tmpData, true);
	    descriptorsVector.col(0).copyTo(trainingData.col(trainingCount));
            fclose(fdata);
            //printf("%s:%d,%d\n",__FUNCTION__,__LINE__,trainingCount);
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

        printf("%s:%d\n",__FUNCTION__,__LINE__);
	svm->save(SVM_OUTPUT_NAME);
        printf("%s:%d\n",__FUNCTION__,__LINE__);

	return true;
}

long GetFileSize(std::string filename)
{
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

bool trainFaultSVM2(String* dataTrainPath, String* labelTrainFile)
{
	printf("Initialize\n");

	// Testing if there are images in the pathes
	if (!dataTrainPath || (*dataTrainPath).length()<=0)
	{
		printf("There are no trainning data %s\n", *dataTrainPath);
		return false;
	}
	if (!labelTrainFile || (*labelTrainFile).length()<=0)
	{
		printf("There are no label file\n");
		return false;
	}

        printf("%s:%d\n",__FUNCTION__,__LINE__);
   
        long sampleSize = GetFileSize(*dataTrainPath)/4;

	Mat trainingLabel = Mat_<int>(1, sampleSize);
	Mat trainingData = Mat_<float>(SEISMIC_SUBVOLUME_SIZE, sampleSize);
	int trainingCount = 0;

        printf("%s:%d\n",__FUNCTION__,__LINE__);

	clock_t beginTime = clock();

        FILE *flabel = fopen((*labelTrainFile).c_str(),"r");
        int labelCount=0;
        while( labelCount<sampleSize && (!feof(flabel)) )
        {
            float v;
            unsigned char temp[4];
            fread((void*)(&temp[0]), sizeof(temp[0]), 1, flabel);
            fread((void*)(&temp[1]), sizeof(temp[1]), 1, flabel);
            fread((void*)(&temp[2]), sizeof(temp[2]), 1, flabel);
            fread((void*)(&temp[3]), sizeof(temp[3]), 1, flabel);
            //int itemp = (temp[0]<<0) | (temp[1]<<8) | (temp[2]<<16) | (temp[3]<<24);
            unsigned int itemp =  (temp[3]<<0) | (temp[2]<<8) | (temp[1]<<16) | (temp[0]<<24);
            v = *((float*)&itemp);

            //printf("%s:%d, %d:%f\n",__FUNCTION__,__LINE__,labelCount,v);
            trainingLabel.at<int>(0, labelCount) = v>0.5?1:-1;
            /*Only for Debug!!! if(labelCount%10==0) trainingLabel.at<int>(0, labelCount) = 1;*/
            labelCount++;
        }
        fclose(flabel);
        printf("%s:%d\n",__FUNCTION__,__LINE__);

        std:vector<float> sampleData(sampleSize);
        FILE *fsample = fopen((*dataTrainPath).c_str(),"r");
        int sampleCount=0;
        while( sampleCount<sampleSize && (!feof(fsample)) )
        {
            float v;
            unsigned char temp[4];
            fread((void*)(&temp[0]), sizeof(temp[0]), 1, flabel);
            fread((void*)(&temp[1]), sizeof(temp[1]), 1, flabel);
            fread((void*)(&temp[2]), sizeof(temp[2]), 1, flabel);
            fread((void*)(&temp[3]), sizeof(temp[3]), 1, flabel);
            //int itemp = (temp[0]<<0) | (temp[1]<<8) | (temp[2]<<16) | (temp[3]<<24);
            unsigned int itemp =  (temp[3]<<0) | (temp[2]<<8) | (temp[1]<<16) | (temp[0]<<24);
            v = *((float*)&itemp);
            sampleData[sampleCount] = v;
            //printf("%s:%d, %d:%f\n",__FUNCTION__,__LINE__,sampleCount,v);
            sampleCount++;
        }
        fclose(fsample);
        /*
        std:vector<float> sampleData(sampleSize);
        FILE *fsample = fopen((*dataTrainPath).c_str(),"r");
        fread(&sampleData[0], sampleSize*4, 1, fsample);
        fclose(fsample);
        */

        int n1=101,n2=102,n3=103;
        int sub3=3,sub2=3,sub1=3;
        int half3=1,half2=1,half1=1;

        int acc=0;
        int finished=0;
        for(int i3=0;i3<n3;i3++)
        {
            if(finished>0) break;
            for(int i2=0;i2<n2;i2++)
            {
                if(finished>0) break;
                for(int i1=0;i1<n1;i1++)
                {
                    int start1 = i1 - half1;
                    int start2 = i2 - half2;
                    int start3 = i3 - half3;
                    std::vector<float> tmpData(SEISMIC_SUBVOLUME_SIZE);
                    int complete=0;
                    for(int j3=0;j3<sub3;j3++)
                    {
                        for(int j2=0;j2<sub2;j2++)
                        {
                            for(int j1=0;j1<sub1;j1++)
                            {
                                int tmp1 = start1+i1;
                                int tmp2 = start2+i2;
                                int tmp3 = start3+i3;
                                if((tmp1>=0&&tmp1<n1)&&(tmp2>=0&&tmp2<n2)&&(tmp3>=0&&tmp3<n3))
                                {
                                    tmpData[j3*(sub2*sub1)+j2*sub1+j1] = sampleData[tmp3*(n2*n1)+tmp2*n1+tmp1];
                                    complete++;
                                } else {
                                    tmpData[j3*(sub2*sub1)+j2*sub1+j1] = -1;
                                }
                            }
                        }
                    }
                    acc++;
                    //if(complete=(sub3*sub2*sub1-1))
                    {
                        Mat descriptorsVector = Mat_<float>(tmpData, true);
                        descriptorsVector.col(0).copyTo(trainingData.col(trainingCount));
                    }
                    printf("%s:%d,sample data:%d,%f\n",__FUNCTION__,__LINE__,acc,sampleData[i3*(n2*n1)+i2*n1+i1]);
                    if(acc>=(n1*n2*n3-1)) {
                        finished = 1;
                        break;
                    }
                }
            }
            printf("%s:%d,Finish image:%d\n",__FUNCTION__,__LINE__,n3);
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

        printf("%s:%d\n",__FUNCTION__,__LINE__);
	svm->save(SVM_OUTPUT_NAME);
        printf("%s:%d\n",__FUNCTION__,__LINE__);

	return true;
}


bool trainFaultSVM3(String* dataTrainPath, String* labelTrainFile)
{
	printf("Initialize\n");
        std::vector<String> dataFileNames;
	// Testing if there are images in the pathes
	if (!dataTrainPath || (*dataTrainPath).length()<=0)
	{
		printf("There are no trainning data %s\n", *dataTrainPath);
		return false;
	}
	if (!labelTrainFile || (*labelTrainFile).length()<=0)
	{
		printf("There are no label file\n");
		return false;
	}

        glob(*dataTrainPath, dataFileNames);

        printf("%s:%d\n",__FUNCTION__,__LINE__);
   
        long sampleSize = GetFileSize(*labelTrainFile)/4;

	Mat trainingLabel = Mat_<int>(1, sampleSize);
	Mat trainingData = Mat_<float>(dataFileNames.size(), sampleSize);
	int trainingCount = 0;

        printf("%s:%d\n",__FUNCTION__,__LINE__);

	clock_t beginTime = clock();

        FILE *flabel = fopen((*labelTrainFile).c_str(),"r");
        int labelCount=0;
        while( labelCount<sampleSize && (!feof(flabel)) )
        {
            float v;
            unsigned char temp[4];
            fread((void*)(&temp[0]), sizeof(temp[0]), 1, flabel);
            fread((void*)(&temp[1]), sizeof(temp[1]), 1, flabel);
            fread((void*)(&temp[2]), sizeof(temp[2]), 1, flabel);
            fread((void*)(&temp[3]), sizeof(temp[3]), 1, flabel);
            //int itemp = (temp[0]<<0) | (temp[1]<<8) | (temp[2]<<16) | (temp[3]<<24);
            unsigned int itemp =  (temp[3]<<0) | (temp[2]<<8) | (temp[1]<<16) | (temp[0]<<24);
            v = *((float*)&itemp);

            //printf("%s:%d, %d:%f\n",__FUNCTION__,__LINE__,labelCount,v);
            trainingLabel.at<int>(0, labelCount) = v>0.5?1:-1;
            /*Only for Debug!!! if(labelCount%10==0) trainingLabel.at<int>(0, labelCount) = 1;*/
            labelCount++;
        }
        fclose(flabel);
        printf("%s:%d\n",__FUNCTION__,__LINE__);
        int fileIdx = 0;
	for (std::vector<String>::iterator fileName = dataFileNames.begin(); fileName != dataFileNames.end(); ++fileName)
	{
            //std:vector<float> sampleData(sampleSize);
            FILE *fsample = fopen((*fileName).c_str(),"r");
            int sampleCount=0;
            while( sampleCount<sampleSize && (!feof(fsample)) )
            {
                float v;
                unsigned char temp[4];
                fread((void*)(&temp[0]), sizeof(temp[0]), 1, flabel);
                fread((void*)(&temp[1]), sizeof(temp[1]), 1, flabel);
                fread((void*)(&temp[2]), sizeof(temp[2]), 1, flabel);
                fread((void*)(&temp[3]), sizeof(temp[3]), 1, flabel);
                //int itemp = (temp[0]<<0) | (temp[1]<<8) | (temp[2]<<16) | (temp[3]<<24);
                unsigned int itemp =  (temp[3]<<0) | (temp[2]<<8) | (temp[1]<<16) | (temp[0]<<24);
                v = *((float*)&itemp);
                //sampleData[sampleCount] = v;
                printf("%s:%d, file:%d,sample:%d:%f vs: %d\n",__FUNCTION__,__LINE__,fileIdx,sampleCount,v,trainingLabel.at<int>(0, sampleCount));
                trainingData.at<float>(fileIdx,sampleCount) = v;
                sampleCount++;
            }
            fclose(fsample);
            fileIdx++;
        }

        /*
        std:vector<float> sampleData(sampleSize);
        FILE *fsample = fopen((*dataTrainPath).c_str(),"r");
        fread(&sampleData[0], sampleSize*4, 1, fsample);
        fclose(fsample);
        */

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

        printf("%s:%d\n",__FUNCTION__,__LINE__);
	svm->save(SVM_OUTPUT_NAME);
        printf("%s:%d\n",__FUNCTION__,__LINE__);

	return true;
}

