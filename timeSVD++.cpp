#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <vector>
#include <cmath>
#include <stack>
#include <sstream>
#include <random>
#include <algorithm>
#include <chrono>

using namespace std;

string directory; 

const int userCount = 458293;
const int movieCount = 17770;

const double avgRating = 3.6033;
int k = 50;
int epochs = 200;
double l = 0.005;
double reg = 0.007;
double reg2 = 0.005;

const int maxRatings = 4000; 

const int num_ratings = 98291669;
vector<vector <int> > ratings(num_ratings);

double* Bi = new double[movieCount];
double* Bu = new double[userCount];

double** Pu = new double*[userCount];
double** Qi = new double*[movieCount];

double** Yj = new double*[movieCount];

double* trainErrors = new double[num_ratings];

short** neighbors = new short*[userCount];
double* Nu = new double[userCount];

int probe_size = 1374739;
int** probe;

string basename = "E:\\Documents\\Caltech\\CS156\\base.dta";
string qualname = "E:\\Documents\\Caltech\\CS156\\qual.dta";
string probename = "E:\\Documents\\Caltech\\CS156\\probe.dta";
string solname = "E:\\Documents\\Caltech\\CS156\\50\\solution";
string probesolname = "E:\\Documents\\Caltech\\CS156\\50\\probesolution";

void initialize_ratings()
{
	ifstream file(basename);
	string input;
	int user, movie, time, rating;
	int index = 0;
	int prevMovie = 0;

	while (getline(file, input))
	{
		sscanf(input.c_str(), "%d %d %d %d", &user, &movie, &time, &rating);

		vector<int> curr(3);
		curr[0] = user - 1;
		curr[1] = movie - 1;
		curr[2] = rating; 

		ratings[index] = curr;
		trainErrors[index] = 0; 

		index++;

		if (movie != prevMovie && movie % 1000 == 0)
		{
			std::cout << "Loading: " << movie << "/" << movieCount << std::endl;
			prevMovie = movie;
		}
	}

	file.close();

	probe = new int*[probe_size];

	ifstream probefile(probename);
	index = 0;

	while (getline(probefile, input))
	{
		sscanf(input.c_str(), "%d %d %d %d", &user, &movie, &time, &rating);

		probe[index] = new int[3];
		probe[index][0] = user - 1;
		probe[index][1] = movie - 1;
		probe[index][2] = rating;

		index++;
	}

	probefile.close();
};

void initialize_svd()
{

	// Initialize all arrays with size userCount
	for (int i = 0; i < userCount; i++)
	{
		neighbors[i] = new short[maxRatings];
		for (int j = 0; j < maxRatings; j++)
			neighbors[i][j] = -1; 


		Bu[i] = rand() / RAND_MAX / sqrt(k);

		Pu[i] = new double[k];
		for (int j = 0; j < k; j++)
			Pu[i][j] = rand() / RAND_MAX / sqrt(k);

	}

	// Initialize all arrays with size movieCount
	for (int i = 0; i < movieCount; i++)
	{
		Bi[i] = rand() / RAND_MAX / sqrt(k);
		
		Yj[i] = new double[k];
		Qi[i] = new double[k];

		for (int j = 0; j < k; j++)
		{
			Qi[i][j] = rand() / RAND_MAX / sqrt(k);
			Yj[i][j] = rand() / RAND_MAX / sqrt(k); 
		}
	}

	// Initilaize neighbors
	for (int i = 0; i < num_ratings; i++)
	{
		int user = ratings[i][0];
		int movie = ratings[i][1];

		for (int j = 0; j < maxRatings; j++)
		{
			if (neighbors[user][j] == -1)
			{
				neighbors[user][j] = movie;
				break;
			}
		}
	}

	// Add neighbors from qual 
	string input;
	ifstream qualfile(qualname);

	int user, movie, time;
	while (getline(qualfile, input))
	{
		sscanf(input.c_str(), "%d %d %d", &user, &movie, &time);
		movie = movie - 1;
		user = user - 1;

		for (int j = 0; j < maxRatings; j++)
		{
			if (neighbors[user][j] == -1)
			{
				neighbors[user][j] = movie;
				break;
			}
		}
	}

	qualfile.close();
	// Initialize neighborhood sizes 
	for (int i = 0; i < userCount; i++)
	{
		double count = 0;

		for (int j = 0; j < maxRatings; j++)
		{
			if (neighbors[i][j] == -1) break;
			else count += 1;
		}

		Nu[i] = pow(sqrt(count), -0.5);
	}


};

void addScalarToVector(double* a, double b, int l)
{
	for (int i = 0; i < l; i++)
		a[i] += b;
};

double sumOfVector(int* a, int l)
{
	double sum = 0;
	for (int i = 0; i < l; i++)
		sum += a[i];
	return sum;
};

void addVectors(double* a, double* b, int l)
{
	for (int i = 0; i < l; i++)
		a[i] += b[i];
};

void multiplyVectorByScalar(double* a, double b, int l)
{
	for (int i = 0; i < l; i++)
		a[i] = a[i] * b;
};

double magnitude(int* a, int l)
{
	double result = 0;
	for (int i = 0; i < l; i++)
		result += a[i] * a[i];

	result = sqrt(result);
	return result; 
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

int* getNeighbors(int user)
{
	int* implicit = new int[movieCount];

	for (int i = 0; i < movieCount; i++)
		implicit[i] = 0;

	for (int i = 0; i < 3341; i++)
		if (neighbors[user][i] == -1) break; 
		else implicit[neighbors[user][i]] = 1; 

	return implicit;
};

double* userVector(int user)
{
	double mag = Nu[user];

	double* sumYj = new double[k];

	for (int i = 0; i < k; i++)
		sumYj[i] = 0; 

	for (int i = 0; i < maxRatings; i++)
	{
		int curr = neighbors[user][i];
		if (curr == -1) break;
		else addVectors(sumYj, Yj[curr], k);
	}
	
	multiplyVectorByScalar(sumYj, mag, k); 
	 
	double* p = Pu[user];
	addVectors(sumYj, p, k); 

	return sumYj; 
};

double predict(int user, int movie)
{
	double bu = Bu[user];
	double bi = Bi[movie];

	double* p = userVector(user);

	double product = dot(p, Qi[movie], 0, k, 0);

	double result = avgRating + bu + bi + product;

	delete p;

	return result; 
};

double trainRMSE()
{
	double sum = 0;

	for (int i = 0; i < num_ratings; i++)
	{
		sum += pow(trainErrors[i], 2);
	}

	double RMSE = sqrt(sum / num_ratings);
	return RMSE;
};

double probeRMSE()
{
	double sum = 0;

	for (int i = 0; i < probe_size; i++)
	{
		int user = probe[i][0];
		int movie = probe[i][1];
		int rating = probe[i][2];
		sum += pow(rating - predict(user, movie), 2);
	}

	double RMSE = sqrt(sum / probe_size);
	return RMSE;
};

void generate_qual(int num)
{
	string input;
	ifstream qualfile(qualname);

	string filename = solname + std::to_string(num) + ".dta";
	cout << filename << endl;
	ofstream solution(filename);

	int user, movie, time;
	while (getline(qualfile, input))
	{
		sscanf(input.c_str(), "%d %d %d", &user, &movie, &time);
		movie = movie - 1;
		user = user - 1;

		double prediction = predict(user, movie);
		if (prediction > 5) prediction = 5;
		if (prediction < 1) prediction = 1; 

		solution << prediction << std::endl;
	}

	qualfile.close();
	solution.close();
};

void generate_probesols(int num)
{
	string filename = probesolname + std::to_string(num) + ".dta";
	ofstream solution(filename);

	for (int i = 0; i < probe_size; i++)
	{
		int user = probe[i][0];
		int movie = probe[i][1];
		double prediction = predict(user, movie);
		if (prediction > 5) prediction = 5;
		if (prediction < 1) prediction = 1;

		solution << prediction << std::endl;
	}

	solution.close();
};

void shuffleRatings()
{
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(std::begin(ratings), std::end(ratings), g);
};

void train_svd()
{
	for (int e = 0; e < epochs; e++)
	{
		std::cout << "Starting Epoch : " << e << std::endl;
		std::cout << "Shuffling... " << std::endl;
		
		chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();

		for (int i = 0; i < num_ratings; i++)
		{
			int user = ratings[i][0];
			int movie = ratings[i][1];
			int rating = ratings[i][2];
			
			// Replicate uvector calculation
			double mag = Nu[user];

			double* uvector = new double[k];

			for (int i = 0; i < k; i++)
				uvector[i] = 0;

			for (int i = 0; i < maxRatings; i++)
			{
				int curr = neighbors[user][i];
				if (curr == -1) break;
				else addVectors(uvector, Yj[curr], k);
			}

			multiplyVectorByScalar(uvector, mag, k);

			double* pu = Pu[user];
			addVectors(uvector, pu, k);

			// Continue 

			// Replicate prediction calculation 
			double bu = Bu[user];
			double bi = Bi[movie];

			double product = dot(uvector, Qi[movie], 0, k, 0);
			double prediction = avgRating + bu + bi + product;
			// Continue 
			
			double err = rating - prediction; 
			Bu[user] += reg * (err - reg2 * bu);
			Bi[movie] += reg * (err - reg2 * bi);

			double* qi = Qi[movie];

			for (int j = 0; j < maxRatings; j++)
			{
				int index = neighbors[user][j];

				if (index == -1) break; 

				for (int z = 0; z < k; z++)
					Yj[index][z] += reg * (err * mag * qi[z] - reg2 * Yj[index][z]);
			}

			for (int j = 0; j < k; j++)
			{
				Pu[user][j] += reg * (err * qi[j] - reg2 * pu[j]);
				Qi[movie][j] += reg * (err * uvector[j] - reg2 * qi[j]);
			}

			if (i % 1000000 == 0)
			{
				std::cout << "Pass:  " << i;
				std::cout << ", Prediction:  " << prediction;
				std::cout << ", Err:  " << err << std::endl;
			}

			trainErrors[i] = err; 

			delete uvector; 
		}

		std::cout << "FINISHED EPOCH:  " << e << std::endl;
		std::cout << "Train RMSE:  " << trainRMSE() << std::endl;
		std::cout << "Probe RMSE:  " << probeRMSE() << std::endl;

		chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
		auto duration = chrono::duration_cast<chrono::minutes>(t2 - t1).count();
		cout << "Time elapsed: " << duration << endl;

		generate_qual(e);
		generate_probesols(e);
	}
};


int main()
{
	/*cout << "Enter directory: " << endl;
	std::cin >> directory; 

	cout << "Enter number of factors (default 50): " << endl;
	std::cin >> k; 

	cout << "Enter number of epochs (default 30): " << endl;
	std::cin >> epochs;*/

	std::cout << "Started..." << std::endl;

	initialize_ratings();
	std::cout << "Finished reading rating data..." << std::endl;

	initialize_svd();
	std::cout << "Initialized SVD..." << std::endl;

	std::cout << "Training SVD... " << std::endl;
	train_svd();

	std::cout << "Done. ";

	std::cin.get();
	return 0;
}
