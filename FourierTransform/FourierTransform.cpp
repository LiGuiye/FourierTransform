#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
using namespace cv;

//傅里叶变换
void dftImage(InputArray _src, OutputArray _dst)
{
	//得到Mat类型
	Mat src = _src.getMat();
	//判断位深
	CV_Assert(src.type() == CV_32FC1 || src.type() == CV_64FC1);
	CV_Assert(src.channels() == 1 || src.channels() == 2);
	int rows = src.rows;
	int cols = src.cols;
	//为了进行快速的傅里叶变换，我们经行和列的扩充,找到最合适扩充值
	Mat padded;
	int rPadded = getOptimalDFTSize(rows);
	int cPadded = getOptimalDFTSize(cols);
	//进行边缘扩充,扩充值为零
	copyMakeBorder(src, padded, 0, rPadded - rows, 0, cPadded - cols, BORDER_CONSTANT, Scalar::all(0));

	//给计算出来的结果分配空间
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) }; //将图像转换成浮点类型
	Mat complexI;
	merge(planes, 2, complexI);         //为延扩后的图像增添一个初始化为0的通道

	dft(complexI, complexI);            //进行离散傅立叶变换. 支持图像原地计算 (输入输出为同一图像):
	dft(padded, _dst, DFT_COMPLEX_OUTPUT); //返回结果进行逆变换（双通道：用于存储实部 和 虚部）；

	// 将复数转换为幅度
	// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude  
	Mat magI = planes[0];

	magI += Scalar::all(1);                    // 切换到对数尺度
	log(magI, magI);

	//剪切和重分布幅度图象限.
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

	// 重新排列傅立叶图像的象限，使原点位于图像中心
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	Mat q0(magI, Rect(0, 0, cx, cy));   // 左上角 - 每个象限创建一个ROI
	Mat q1(magI, Rect(cx, 0, cx, cy));  // 右上
	Mat q2(magI, Rect(0, cy, cx, cy));  // 左下方
	Mat q3(magI, Rect(cx, cy, cx, cy)); // 右下

	Mat tmp;                           // 交换象限（左上角和右上角）
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // 交换象限（右上角与左下角）
	q2.copyTo(q1);
	tmp.copyTo(q2);
	
	//归一化
	normalize(magI, magI, 0, 1, CV_MINMAX); // 使用浮点值将矩阵转换为可视图像形式（在值0和1之间浮动）。
	
	namedWindow("DFT结果", WINDOW_AUTOSIZE);
	imshow("DFT结果", magI);

}
int main(int argc, char*argv[])
{
	//输入图像矩阵
	Mat img = imread("../data/lady.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	if (!img.data)
		return -1;
	//数据类型转换：转换为浮点型
	Mat fImg;
	img.convertTo(fImg, CV_64FC1);
	//傅里叶变换
	Mat dft1;
	dftImage(fImg, dft1);

	//傅里叶逆变换
	Mat image;
	cv::dft(dft1, image, DFT_INVERSE + DFT_REAL_OUTPUT + DFT_SCALE);
	//裁剪傅里叶逆变换
	image = image(Rect(0, 0, img.cols, img.rows));
	image.convertTo(image, CV_8UC1);

	namedWindow("原图", WINDOW_AUTOSIZE);
	imshow("原图", img);

	namedWindow("逆变换图", WINDOW_AUTOSIZE);
	imshow("逆变换图", image);

	waitKey(0);

	return 0;

}