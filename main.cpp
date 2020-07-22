#define PI 3.14159265359

#include "util.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <fstream>
#include <limits.h>
#include <stdint.h>
#include <list>
#include <opencv/cv.h>
#include <opencv2/features2d/features2d.hpp>
#include <Windows.h>
#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <cstdlib>
#include <ctime>

using namespace cv;
using namespace Eigen;
using namespace std;

Matrix<double, 3, 3> makeK_a(void)
{
	Matrix<double, 3, 3> K_a;

	double fx = 637.890687;
	double fy = 637.837657;
	double cx = 305.931644;
	double cy = 220.467440;

	K_a(0, 0) = fx;
	K_a(0, 1) = 0;
	K_a(0, 2) = cx;

	K_a(1, 0) = 0;
	K_a(1, 1) = fy;
	K_a(1, 2) = cy;

	K_a(2, 0) = 0;
	K_a(2, 1) = 0;
	K_a(2, 2) = 1;

	return K_a;

}

Matrix<double, 3, 3> makeK_b(void)
{
	Matrix<double, 3, 3> K_b;

	double fx = 638.923602;
	double fy = 638.832943;
	double cx = 327.234846;
	double cy = 231.953842;

	K_b(0, 0) = fx;
	K_b(0, 1) = 0;
	K_b(0, 2) = cx;

	K_b(1, 0) = 0;
	K_b(1, 1) = fy;
	K_b(1, 2) = cy;

	K_b(2, 0) = 0;
	K_b(2, 1) = 0;
	K_b(2, 2) = 1;

	return K_b;

}



Matrix<double, 4, 4> makeT(double robot_x, double robot_y, double robot_z, double robot_theta, double camera_x, double camera_y, double camera_z, double camera_theta_y, double camera_theta_z)
{
	//double camera_z = 80.57 * 0.001; 


	/* camera pose from robot pose
						3A        2A         1A        0            1B         2B         3B
	x                 217.01    217.01      217.01    217.01       217.01     217.01     217.01
	y                -199.47     -139.47     -79.47     0           79.47      139.47     199.47
	z                  80.57   
	camera_theta        0         -22.5      -45         0           +45         +22.5      +45
	*/

	double th1 = 90 * M_PI / 180;
	double th2 = (90 + robot_theta + camera_theta_z) * M_PI / 180;
	double th3 = camera_theta_y * M_PI / 180;

	Matrix<double, 4, 4> T;
	Matrix<double, 3, 3> Rcw;

	Rcw(0, 0) = cos(th2);
	Rcw(0, 1) = sin(th2)*sin(th3);
	Rcw(0, 2) = cos(th3)*sin(th2);

	Rcw(1, 0) = sin(th1)*sin(th2);
	Rcw(1, 1) = cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3);
	Rcw(1, 2) = -cos(th1)*sin(th3) - cos(th2)*cos(th3)*sin(th1);

	Rcw(2, 0) = -cos(th1)*sin(th2);
	Rcw(2, 1) = cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3);
	Rcw(2, 2) = cos(th1)*cos(th2)*cos(th3) - sin(th1)*sin(th3);

	Matrix<double, 3, 1> tcw;
	Matrix<double, 3, 1> twc;

	tcw(0, 0) = robot_x + (camera_x*cos(robot_theta* M_PI / 180) - camera_y*sin(robot_theta* M_PI / 180));
	tcw(1, 0) = robot_y + (camera_x*sin(robot_theta* M_PI / 180) + camera_y*cos(robot_theta* M_PI / 180));
	tcw(2, 0) = robot_z + camera_z;

	twc = -Rcw.transpose()*tcw;

	//Rcw의 transpose  = Rwc, Rwc이용하여 Twc 생성
	T(0, 0) = Rcw(0, 0);
	T(0, 1) = Rcw(1, 0);
	T(0, 2) = Rcw(2, 0);
	T(0, 3) = twc(0, 0);

	T(1, 0) = Rcw(0, 1);
	T(1, 1) = Rcw(1, 1);
	T(1, 2) = Rcw(2, 1);
	T(1, 3) = twc(1, 0);

	T(2, 0) = Rcw(0, 2);
	T(2, 1) = Rcw(1, 2);
	T(2, 2) = Rcw(2, 2);
	T(2, 3) = twc(2, 0);

	T(3, 0) = 0;
	T(3, 1) = 0;
	T(3, 2) = 0;
	T(3, 3) = 1;

	return T;
}

vector<vector<double>> get_full_feature_world(void)
{

	static const char *pos_dir_feat = "D:\\git\\extrinsic_data\\1\\feature_worldframe.txt";
	FILE *ft = fopen(pos_dir_feat, "r");
	vector<vector<double>> full_feature_world;
	double fx1, fy1, fz1;
	while (!feof(ft))
	{
		vector<double> feature_world;
		fscanf(ft, "%lf\t%lf\t%lf\n", &fx1, &fy1, &fz1);
		feature_world.push_back(fx1 * 0.001);
		feature_world.push_back(fy1 * 0.001);
		feature_world.push_back(fz1 * 0.001);
		//cout << feature_world << endl;
		full_feature_world.push_back(feature_world);
	}
	fclose(ft);

	return full_feature_world;
}

vector<pair<double, double>> get_pixel_camera_ref(static const char *pos_dir_cam)
{
	//static const char *pos_dir_cam= dir;
	FILE *ft = fopen(pos_dir_cam, "r");
	vector<pair<double, double>> pixel_camera_ref;
	double cx1, cy1;

	while (!feof(ft))
	{
		pair<double, double> pixel_cam;
		fscanf(ft, "%lf\t%lf\n", &cx1, &cy1);
		pixel_cam.first = cx1;
		pixel_cam.second = cy1;
		//cout << feature_world << endl;
		pixel_camera_ref.push_back(pixel_cam);
	}
	fclose(ft);

	return pixel_camera_ref;
}

vector<vector<double>> get_robot_world(static const char *pos_dir_robot)
{
	FILE *ft = fopen(pos_dir_robot, "r");
	vector<vector<double>> full_robot_world;
	double rx, ry, rz, tx, ty, tz;

	while (!feof(ft))
	{
		vector<double> robot_world;
		fscanf(ft, "%lf\t%lf\t%lf\t%lf\n", &rz, &tx, &ty, &tz);

		robot_world.push_back(rz * 180 / M_PI); //rz 단위 변환
		robot_world.push_back(tx * 0.001); //[m]
		robot_world.push_back(ty * 0.001);
		robot_world.push_back(tz * 0.001);

		full_robot_world.push_back(robot_world);
	}

	fclose(ft);
	return full_robot_world;
}

int main(void)
{
	//data 받기
	vector<vector<double>> full_feature_world;
	full_feature_world = get_full_feature_world();
	//full_feature_world[0~14][0]:world 좌표 기준 feature x좌표
	//full_feature_world[0~14][1]:world 좌표 기준 feature y좌표
	//full_feature_world[0~14][2]:world 좌표 기준 feature z좌표

	vector<pair<double, double>> pixel_camera_ref_0A;
	vector<pair<double, double>> pixel_camera_ref_1A;
	vector<pair<double, double>> pixel_camera_ref_1B;
	vector<pair<double, double>> pixel_camera_ref_2A;
	vector<pair<double, double>> pixel_camera_ref_2B;
	vector<pair<double, double>> pixel_camera_ref_3A;
	vector<pair<double, double>> pixel_camera_ref_3B;
	//pixel_camera_ref_0A = get_pixel_camera_ref("D:\\git\\extrinsic_data\\2\\pixel_camera_ref_0A.txt");
	//pixel_camera_ref_0A, pixel_camera_ref_0A[0~14].first, .second (x,y)
	pixel_camera_ref_1A = get_pixel_camera_ref("D:\\git\\extrinsic_data\\1\\pixel_camera_ref_1A.txt");
	pixel_camera_ref_1B = get_pixel_camera_ref("D:\\git\\extrinsic_data\\1\\pixel_camera_ref_1B.txt");
	pixel_camera_ref_2A = get_pixel_camera_ref("D:\\git\\extrinsic_data\\1\\pixel_camera_ref_2A.txt");
	pixel_camera_ref_2B = get_pixel_camera_ref("D:\\git\\extrinsic_data\\1\\pixel_camera_ref_2B.txt");
	pixel_camera_ref_3A = get_pixel_camera_ref("D:\\git\\extrinsic_data\\1\\pixel_camera_ref_3A.txt");
	pixel_camera_ref_3B = get_pixel_camera_ref("D:\\git\\extrinsic_data\\1\\pixel_camera_ref_3B.txt");

	vector<vector<double>> get_robot_world_0A;
	vector<vector<double>> get_robot_world_1A;
	vector<vector<double>> get_robot_world_1B;
	vector<vector<double>> get_robot_world_2A;
	vector<vector<double>> get_robot_world_2B;
	vector<vector<double>> get_robot_world_3AB;
	//get_robot_world_0A = get_robot_world("D:\\git\\extrinsic_data\\2\\robot_world_0A.txt");
	//get_robot_world_0A[0~14][0,1,2,3]
	get_robot_world_1A = get_robot_world("D:\\git\\extrinsic_data\\1\\robot_world_1A.txt");
	get_robot_world_1B = get_robot_world("D:\\git\\extrinsic_data\\1\\robot_world_1B.txt");
	get_robot_world_2A = get_robot_world("D:\\git\\extrinsic_data\\1\\robot_world_2A.txt");
	get_robot_world_2B = get_robot_world("D:\\git\\extrinsic_data\\1\\robot_world_2B.txt");
	get_robot_world_3AB = get_robot_world("D:\\git\\extrinsic_data\\1\\robot_world_3AB.txt");

	//Intrinsic Matrix 생성
	Matrix<double, 3, 3> K_a;
	K_a = makeK_a();
	Matrix<double, 3, 3> K_b;
	K_b = makeK_b();

	/* camera pose from robot pose
						3A        2A         1A        0            1B         2B         3B
	x				  223.88
	y                -199.47     -139.47     -79.47     0           79.47      139.47     199.47
	z                  81.99
	camera_theta        0         -22.5      -45.0         0           +45.0        +22.5      0
	*/

	vector<vector<double>> error_3B;
	for (double i = -70.0; i <=-68.0; i = i + 1.0) 
	{
		for (double j = 15.0; j <= 17.0; j = j + 1.0)
		{
			for (double l = -45.0; l <= -43.0; l = l + 1.0)
			{
				//cout <<"1B  " <<  "i : " << i << " j : " << j << " l : " << l << endl;
				for (double m = -2.0; m <= 0.0; m = m + 1.0)
				{
					for (double n = -4.0; n <= -2.0; n = n + 1.0)
					{
						double error_3B_sum = 0;
						vector<double> comp_error_n_pos;

						double camera_x = 0.16088;// (223.88 + i) *0.001;
						double camera_y = 0.19247;// (79.47 + j) *0.001;
						double camera_z = 0.04199;// (81.99 + l) *0.001;
						double camera_theta_y = 2.0;// m; //degree
						double camera_theta_z = 0.0;// (45.0 + n);  //degree

						for (int r = 0; r < 15; r++) //robot
						{
							if (r == 7)
								break;
							else
							{
								double robot_x = get_robot_world_3AB[r][1]; //[m]
								double robot_y = get_robot_world_3AB[r][2]; //[m]
								double robot_z = get_robot_world_3AB[r][3]; //[m]
								double robot_theta = get_robot_world_3AB[r][0]; //degree

								Matrix<double, 4, 4> Twc;
								Twc = makeT(robot_x, robot_y, robot_z, robot_theta, camera_x, camera_y, camera_z, camera_theta_y, camera_theta_z);

								for (int k = 0; k < 15; k++) //feature point
								{
									if (pixel_camera_ref_3B[15 * r + k].first != -1)
									{
										//feature의 world좌표
										Matrix<double, 4, 1> world_XYZ;
										world_XYZ(0, 0) = full_feature_world[k][0];
										world_XYZ(1, 0) = full_feature_world[k][1];
										world_XYZ(2, 0) = full_feature_world[k][2];
										world_XYZ(3, 0) = 1;

										//[u v 1]' = K[R|t][X Y Z 1]' 계산과정
										Matrix<double, 4, 1> result_sub;
										result_sub = Twc*world_XYZ;

										Matrix<double, 3, 1> result_sub_new;
										result_sub_new(0, 0) = result_sub(0, 0) / result_sub(3, 0);
										result_sub_new(1, 0) = result_sub(1, 0) / result_sub(3, 0);
										result_sub_new(2, 0) = result_sub(2, 0) / result_sub(3, 0);

										Matrix<double, 3, 1> uv_1;
										//uv_1 = K_a*result_sub_new;
										uv_1 = K_b*result_sub_new;

										Matrix<double, 3, 1> result_pixel;
										result_pixel(0, 0) = uv_1(0, 0) / uv_1(2, 0);
										result_pixel(1, 0) = uv_1(1, 0) / uv_1(2, 0);
										result_pixel(2, 0) = uv_1(2, 0) / uv_1(2, 0); // = 1

										result_pixel(0, 0) = 640 - result_pixel(0, 0);
										result_pixel(1, 0) = 480 - result_pixel(1, 0);

										error_3B_sum = error_3B_sum + sqrt(pow((result_pixel(0, 0) - pixel_camera_ref_3B[15 * r + k].first), 2) + pow((result_pixel(1, 0) - pixel_camera_ref_3B[15 * r + k].second), 2));

									}
									else
										break;
								}
							}
						}

						comp_error_n_pos.push_back(error_3B_sum);
						comp_error_n_pos.push_back(camera_x);
						comp_error_n_pos.push_back(camera_y);
						comp_error_n_pos.push_back(camera_z);
						comp_error_n_pos.push_back(camera_theta_y);
						comp_error_n_pos.push_back(camera_theta_z);

						FILE *pFile = fopen("Resulting_DATA_3B_new.txt", "a");
						fprintf(pFile, "%lf\t %lf\t %lf\t %lf\t %lf\t %lf\n", error_3B_sum, camera_x, camera_y, camera_z, camera_theta_y, camera_theta_z);
						fclose(pFile);

						error_3B.push_back(comp_error_n_pos);
					}
				}
			}
		}
	}

	double min_error_sum;
	int min_index;
	int aa = 0;

	for (int i = 0; i < error_3B.size(); i++)
	{
		if (aa == 0)
		{
			aa = 1;
			min_error_sum = error_3B[i][0];
			min_index = i;
		}
		else
		{
			if (error_3B[i][0] < min_error_sum)
			{
				min_error_sum = error_3B[i][0];
				min_index = i;
			}
		}
	}

	FILE *pFile = fopen("Result_DATA_3B_new.txt", "a");
	fprintf(pFile, "%lf\t %lf\t %lf\t %lf\t %lf\t %lf\n", min_error_sum, error_3B[min_index][1], error_3B[min_index][2], error_3B[min_index][3], error_3B[min_index][4], error_3B[min_index][5]);
	fclose(pFile);

	return 0;

}