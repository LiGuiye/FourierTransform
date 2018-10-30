#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
using namespace cv;

//����Ҷ�任
void dftImage(InputArray _src, OutputArray _dst)
{
	//�õ�Mat����
	Mat src = _src.getMat();
	//�ж�λ��
	CV_Assert(src.type() == CV_32FC1 || src.type() == CV_64FC1);
	CV_Assert(src.channels() == 1 || src.channels() == 2);
	int rows = src.rows;
	int cols = src.cols;
	//Ϊ�˽��п��ٵĸ���Ҷ�任�����Ǿ��к��е�����,�ҵ����������ֵ
	Mat padded;
	int rPadded = getOptimalDFTSize(rows);
	int cPadded = getOptimalDFTSize(cols);
	//���б�Ե����,����ֵΪ��
	copyMakeBorder(src, padded, 0, rPadded - rows, 0, cPadded - cols, BORDER_CONSTANT, Scalar::all(0));

	//����������Ľ������ռ�
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) }; //��ͼ��ת���ɸ�������
	Mat complexI;
	merge(planes, 2, complexI);         //Ϊ�������ͼ������һ����ʼ��Ϊ0��ͨ��

	dft(complexI, complexI);            //������ɢ����Ҷ�任. ֧��ͼ��ԭ�ؼ��� (�������Ϊͬһͼ��):
	dft(padded, _dst, DFT_COMPLEX_OUTPUT); //���ؽ��������任��˫ͨ�������ڴ洢ʵ�� �� �鲿����

	// ������ת��Ϊ����
	// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude  
	Mat magI = planes[0];

	magI += Scalar::all(1);                    // �л��������߶�
	log(magI, magI);

	//���к��طֲ�����ͼ����.
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

	// �������и���Ҷͼ������ޣ�ʹԭ��λ��ͼ������
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	Mat q0(magI, Rect(0, 0, cx, cy));   // ���Ͻ� - ÿ�����޴���һ��ROI
	Mat q1(magI, Rect(cx, 0, cx, cy));  // ����
	Mat q2(magI, Rect(0, cy, cx, cy));  // ���·�
	Mat q3(magI, Rect(cx, cy, cx, cy)); // ����

	Mat tmp;                           // �������ޣ����ϽǺ����Ͻǣ�
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // �������ޣ����Ͻ������½ǣ�
	q2.copyTo(q1);
	tmp.copyTo(q2);
	
	//��һ��
	normalize(magI, magI, 0, 1, CV_MINMAX); // ʹ�ø���ֵ������ת��Ϊ����ͼ����ʽ����ֵ0��1֮�両������
	
	namedWindow("DFT���", WINDOW_AUTOSIZE);
	imshow("DFT���", magI);

}
int main(int argc, char*argv[])
{
	//����ͼ�����
	Mat img = imread("../data/lady.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	if (!img.data)
		return -1;
	//��������ת����ת��Ϊ������
	Mat fImg;
	img.convertTo(fImg, CV_64FC1);
	//����Ҷ�任
	Mat dft1;
	dftImage(fImg, dft1);

	//����Ҷ��任
	Mat image;
	cv::dft(dft1, image, DFT_INVERSE + DFT_REAL_OUTPUT + DFT_SCALE);
	//�ü�����Ҷ��任
	image = image(Rect(0, 0, img.cols, img.rows));
	image.convertTo(image, CV_8UC1);

	namedWindow("ԭͼ", WINDOW_AUTOSIZE);
	imshow("ԭͼ", img);

	namedWindow("��任ͼ", WINDOW_AUTOSIZE);
	imshow("��任ͼ", image);

	waitKey(0);

	return 0;

}