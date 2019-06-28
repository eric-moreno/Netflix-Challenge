// timeSVD.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/SVD>

using namespace std;
using namespace Eigen;

const int userCount = 458293;
const int movieCount = 1000;
const double avgRating = 3.60073; 
const int k = 40;

int main()
{
	std::cout << "Started..." << std::endl;

	// Initialize R 
	double** R = new double*[userCount];

	for (size_t i = 0; i < userCount; i++)
	{
		R[i] = new double[movieCount];
	}

	for (int i = 0; i < userCount; i++)
	{
		for (int j = 0; j < movieCount; j++)
		{
			R[i][j] = -1.0; 
		}
	}
	
	// Read input file all.dta
	ifstream file("E:\\Documents\\Caltech\\CS156\\base.dta");
	string input;
	int user, movie, time, rating;
	while(getline(file, input))
	{
		sscanf_s(input.c_str(), "%d %d %d %d", &user, &movie, &time, &rating);
		R[user - 1][movie - 1] = rating; 
	}

	file.close();

	std::cout << "Finished reading data... " << std::endl;


	// Now we deal with no ratings (-1 values) 
	double rowAverage[userCount];
	for (int i = 0; i < userCount; i++)
	{
		double sum = 0;
		double count = 0;
		for (int j = 0; j < movieCount; j++)
		{
			if (R[i][j] >= 0)
			{
				sum += R[i][j];
				count += 1;
			}
		}
		rowAverage[i] = sum/count; 
	}

	double colAverage[movieCount];
	for (int i = 0; i < movieCount; i++)
	{
		double sum = 0;
		double count = 0;
		for (int j = 0; j < userCount; j++)
		{
			if (R[j][i] >= 0)
			{
				sum += R[j][i];
				count += 1;
			}
		}
		colAverage[i] = sum/count;
	}

	// Set all missing values to column average
	for (int i = 0; i < movieCount; i++)
	{
		double col = colAverage[i];
		for (int j = 0; j < userCount; j++)
		{
			if (R[j][i] < 0)
			{
				R[j][i] = col; 
			}
		}
	}

	// Subtract row average from all values 
	for (int i = 0; i < userCount; i++)
	{
		double sub = rowAverage[i];
		for (int j = 0; j < movieCount; j++)
		{
			R[i][j] = R[i][j] - sub;
		}
	}

	std::cout << "Creating first matrix..." << std::endl; 

	// Sketchy 
	Eigen::MatrixXd RMatrix(userCount, movieCount);
	for (int i = 0; i < userCount; i++)
		RMatrix.row(i) = Eigen::VectorXd::Map(&R[i][0], movieCount);

	BDCSVD<MatrixXd> svd(RMatrix, ComputeFullV | ComputeFullU);

	Eigen::MatrixXd U = svd.matrixU();
	Eigen::MatrixXd V = svd.matrixV();
	Eigen::MatrixXd S = svd.singularValues();

	// Dimensionality reduction
	U.resize(userCount, k);
	V.resize(k, movieCount);
	S.resize(k, k); 

	Eigen::MatrixXd sqrtS = S.cwiseSqrt();
	Eigen::MatrixXd Pu = U * sqrtS.transpose(); // User associations
	Eigen::MatrixXd Qi = sqrtS * V.transpose(); // Movie associations 

	std::cout << "Completed matrix operations..." << std::endl;

	ifstream qualfile("E:\\Documents\\Caltech\\CS156\\qual.dta");
	ofstream solution("E:\\Documents\\Caltech\\CS156\\solution.dta");

	while (getline(qualfile, input))
	{
		sscanf_s(input.c_str(), "%d %d %d", &user, &movie, &time);
		movie = movie - 1;
		user = user - 1; 
		Eigen::MatrixXd product = Qi.row(movie) * Pu.row(user).transpose();
		double inner = product(0, 0);
		double prediction = avgRating + inner + colAverage[movie] + rowAverage[user];

		solution << prediction << std::endl; 
	}

	qualfile.close();
	solution.close();
	
	std::cout << "Done. ";
	
	std::cin.get();
    return 0;
}

