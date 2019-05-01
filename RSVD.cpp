#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <vector>
#include <cmath>
#include <stack>

using namespace std;

const int userCount = 480189;
const int movieCount = 17770;
const int trainSize = 94362233;
const double avgRating = 3.6033;
const int k = 96;
const int epochs = 250;
double l = 0.001;
double reg = 0.02;
double reg2 = 0.05;

int** ratings;
const int num_ratings = 94362233;

double** Movies; // In form average, bias,  count
double** Users;  // Average, bias, count 

double** Pu = new double*[userCount];
double** Qi = new double*[movieCount];

double* residuals; 
double* residualError; 

int probe_size = 1374739;
int** probe; 
double* probeResiduals;
double* probeResidualError; 

// Used for BASIC predictors 
double w[] = { -474519146.8868519, -474519144.8076477, -474519142.7157141, -474519140.99534047, -474519138.9479845, 0.9639865159988403 };
double intercept = 474519145.27352077;

void initialize_ratings()
{
	ratings = new int*[num_ratings];
	residuals = new double[num_ratings];
	residualError = new double[num_ratings];

	string filename = "E:\\Documents\\Caltech\\CS156\\base.dta";
	ifstream file(filename);
	string input; 
	int user, movie, time, rating;
	int index = 0;
	int prevMovie = 0; 

	while (getline(file, input))
	{
		sscanf(input.c_str(), "%d %d %d %d", &user, &movie, &time, &rating);

		ratings[index] = new int[4];
		ratings[index][0] = user;
		ratings[index][1] = movie;
		ratings[index][2] = time;
		ratings[index][3] = rating; 

		residuals[index] = 0;
		residualError[index] = 0;

		index++;

		if (movie != prevMovie && movie % 1000 == 0)
		{
			std::cout << "Loading: " << movie << "/" << movieCount << std::endl;
			prevMovie = movie; 
		}
		
	}

	file.close();

	probe = new int*[probe_size];
	probeResiduals = new double[probe_size];
	probeResidualError = new double[probe_size];

	filename = "E:\\Documents\\Caltech\\CS156\\probe.dta";
	ifstream probefile(filename);
	index = 0;

	while (getline(probefile, input))
	{
		sscanf(input.c_str(), "%d %d %d %d", &user, &movie, &time, &rating);

		probe[index] = new int[4];
		probe[index][0] = user;
		probe[index][1] = movie;
		probe[index][2] = time;
		probe[index][3] = rating;

		probeResiduals[index] = 0;
		probeResidualError[index] = 0;

		index++;
	}

	probefile.close();
};

void initialize_averages()
{
	string movfilename = "E:\\Documents\\Caltech\\CS156\\movies.dta";
	string usrfilename = "E:\\Documents\\Caltech\\CS156\\users.dta";
	string input;

	// Movie averages
	Movies = new double*[movieCount];

	for (int i = 0; i < movieCount; i++)
	{
		Movies[i] = new double[3];
		Movies[i][0] = avgRating;
		Movies[i][1] = 0;
		Movies[i][2] = 1;
	}


	ifstream mfile(movfilename);
	int movie, count;
	double avg, std; 

	while (getline(mfile, input))
	{
		sscanf(input.c_str(), "%d %lf %lf %d", &movie, &avg, &std, &count);
		movie = movie - 1;
		Movies[movie][0] = avg;
		Movies[movie][1] = (avgRating * 25 + avg * count) / (25 + count) - avgRating; // Movie bias
		Movies[movie][2] = count;
	}

	mfile.close();

	// User averages
	Users = new double*[userCount];

	for (int i = 0; i < userCount; i++)
	{
		Users[i] = new double[6];
		Users[i][0] = avgRating;
		Users[i][1] = 0;
		Users[i][2] = 0;
		Users[i][3] = 1;
		Users[i][4] = 0;
		Users[i][5] = 0;
	}

	ifstream userfile("E:\\Documents\\Caltech\\CS156\\usersLinear.dta");
	ifstream userAveragefile("E:\\Documents\\Caltech\\CS156\\users.dta");
	int user;
	double a, b, c, d, e;
	while (getline(userfile, input))
	{
		sscanf(input.c_str(), "%d %lf %lf %lf %lf %lf", &user, &a, &b, &c, &d, &e);

		getline(userAveragefile, input);

		sscanf(input.c_str(), "%d %lf %lf %d", &user, &avg, &std, &count);

		user = user - 1;

		Users[user][0] = avg;
		Users[user][1] = a;
		Users[user][2] = b;
		Users[user][3] = c;
		Users[user][4] = d;
		Users[user][5] = e;
	}

	userfile.close();
	userAveragefile.close();
};

void initialize_svd()
{
	Pu = new double*[userCount];
	Qi = new double*[movieCount];

	for (int i = 0; i < userCount; i++)
	{
		Pu[i] = new double[k];

		for (int j = 0; j < k; j++)
			Pu[i][j] = 0.1;
	}

	for (int i = 0; i < movieCount; i++)
	{
		Qi[i] = new double[k];

		for (int j = 0; j < k; j++)
			Pu[i][j] = 0.1;
	}
};

double clamp(double a, double b)
{
	double sum = a + b;
	if (sum > 5) return 5;
	else if (sum < 1) return 1;
	else return sum;
};

double dot(double a[], double b[], int s, int f, double prev)
{
	double result = prev;
	for (int i = s; i < f; i++)
	{
		double curr = a[i] * b[i];
		result += curr; 
	}
	return result;
};

double baseline(int user, int movie)
{
	double base = (w[0] * Users[user][1]) + (w[1] * Users[user][2]) + (w[2] * Users[user][3])
		+ (w[3] * Users[user][4]) + (w[4] * Users[user][5]) + (w[5] * (Movies[movie][0] - Users[user][0]))
		+ intercept;

	return base; 
}

double predict(int user, int movie)
{
	return baseline(user, movie) + dot(Pu[user], Qi[movie], 0, k, 0);
};

double testRMSE()
{
	double sum = 0; 

	for (int i = 0; i < num_ratings; i++)
	{
		sum += pow(residualError[i], 2);
	}

	double RMSE = sqrt(sum / num_ratings); 
	return RMSE;
};

double probeRMSE()
{
	double sum = 0;

	for (int i = 0; i < probe_size; i++)
	{
		sum += pow(probeResidualError[i], 2);
	}

	double RMSE = sqrt(sum / probe_size);
	return RMSE;
};


void calculate_residuals(int factor)
{
	for (int j = 0; j < num_ratings; j++)
	{
		int user = ratings[j][0] - 1;
		int movie = ratings[j][1] - 1;

		if (factor == 0)
		{
			double res = clamp(0, predict(user, movie));
			res += -(Pu[user][0] * Qi[movie][0]);
			residuals[j] = res;
		}
		else
		{
			residuals[j] = clamp(residuals[j], (Pu[user][factor - 1] * Qi[movie][factor - 1])); // Add previous dot product
			residuals[j] += -(Pu[user][factor] * Qi[movie][factor]); // Subtract next dot product 
		}
	}

	for (int j = 0; j < probe_size; j++)
	{
		int user = probe[j][0] - 1;
		int movie = probe[j][1] - 1;

		if (factor == 0)
		{
			double res = clamp(0, predict(user, movie));
			res += -(Pu[user][0] * Qi[movie][0]);
			probeResiduals[j] = res;
		}
		else
		{
			probeResiduals[j] = clamp(probeResiduals[j], (Pu[user][factor - 1] * Qi[movie][factor - 1])); // Add previous dot product
			probeResiduals[j] += -(Pu[user][factor] * Qi[movie][factor]); // Subtract next dot product 
		}
	}
};

double predict_with_residual(int index, double next, bool test)
{
	double res;

	if (test)
		res = probeResiduals[index];
	else
		res = residuals[index];

	return res + next; 
};

double vector_average(vector <double> a)
{
	double sum = 0;
	for (int i = 0; i < a.size(); i++)
	{
		sum += a[i];
	}

	return sum / (a.size());
};

void train_svd()
{
	for (int i = 0; i < k; i++)
	{
		calculate_residuals(i);
		std::cout << "Starting feature: " << i << std::endl;

		//double lastRMSE = 10000;

		vector <double> last5;
		vector <double> next5; 

		for (int e = 0; e < epochs; e++)
		{
			// Train set
			for (int j = 0; j < num_ratings; j++)
			{
				int user = ratings[j][0] - 1;
				int movie = ratings[j][1] - 1;
				int rating = ratings[j][3];

				double uv = Pu[user][i];
				double mv = Qi[movie][i];

				double prediction = predict_with_residual(j, uv * mv, false);
				double err = rating - prediction;

				Pu[user][i] = uv + l * (err * mv - reg * uv);
				Qi[movie][i] = mv + l * (err * uv - reg * mv);
				residualError[j] = err; 
			}

			// Probe set 
			for (int j = 0; j < probe_size; j++)
			{
				int user = probe[j][0] - 1;
				int movie = probe[j][1] - 1;
				int rating = probe[j][3];

				double uv = Pu[user][i];
				double mv = Qi[movie][i];

				double prediction = predict_with_residual(j, uv * mv, true);
				double err = rating - prediction;

				probeResidualError[j] = err;
			}

			double currRMSE = testRMSE();
			double testError = probeRMSE();

			if (e < 5) last5.push_back(testError);
			else if (e < 10) next5.push_back(testError);
			else
			{
				//std::cout << "Last5: " << vector_average(last5) << std::endl;
				//std::cout << "Next5: " << vector_average(next5) << std::endl;

				if (vector_average(last5) < vector_average(next5)) break;
				else
				{
					last5.erase(last5.begin());
					last5.push_back(next5.back());
					next5.erase(next5.begin());
					next5.push_back(testError);
				}	
			}

			std::cout << "Feature: " << i << ", Epoch: " << e << std::endl;
			std::cout << "Probe RMSE: " << probeRMSE() << std::endl;
			std::cout << "Train RMSE: " << testRMSE() << std::endl;
			
		}

		std::cout << "Finished feature: " << i << std::endl;
	}
};

void generate_qual()
{
	string input; 
	ifstream qualfile("E:\\Documents\\Caltech\\CS156\\qual.dta");
	ofstream solution("E:\\Documents\\Caltech\\CS156\\solution.dta");

	int user, movie, time; 
	while (getline(qualfile, input))
	{
		sscanf_s(input.c_str(), "%d %d %d", &user, &movie, &time);
		movie = movie - 1;
		user = user - 1;

		double prediction = predict(user, movie);
		prediction = clamp(0, prediction);

		solution << prediction << std::endl;
	}

	qualfile.close();
	solution.close();
}

int main()
{
	std::cout << "Started..." << std::endl;

	initialize_ratings();
	std::cout << "Finished reading rating data..." << std::endl;

	initialize_averages();
	std::cout << "Finished reading averages data..." << std::endl;

	initialize_svd();
	std::cout << "Initialized SVD..." << std::endl;


	std::cout << "Training SVD... " << std::endl;
	train_svd();

	std::cout << "SVD complete, generating solutions..." << std::endl;
	generate_qual();

	std::cout << "Done. ";

	std::cin.get();
	return 0;
}

