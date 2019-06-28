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
#include <map>

using namespace std;

string directory; 

const int userCount = 458293;
const int movieCount = 17770;

const double avgRating = 3.6033;
int k = 1200;
int epochs = 91;

double reg = 0.005;
double reg2 = 0.005;

const int num_ratings = 98291669;
vector<vector <int> > ratings(num_ratings);

double* Bi = new double[movieCount];
double* Bu = new double[userCount];

double** Pu = new double*[userCount];
double** Qi = new double*[movieCount];

double** Yj = new double*[movieCount];

vector<vector <short> > neighbors(userCount);
double* Nu = new double[userCount];

// Time components
int bin_num = 30; 
int max_time = 2243; 
double* Tu = new double[userCount];
double* Alpha_u = new double[userCount];
double** Bi_bin = new double*[movieCount];
vector<map<int, double> > Bu_t(userCount);

vector<vector <int> > userIndices(userCount);

int probe_size = 1374739;
int** probe;

int qual_size = 2749898;
int** qual = new int*[qual_size];

string basename = "E:\\Documents\\Caltech\\CS156\\umbase.dta";
string qualname = "E:\\Documents\\Caltech\\CS156\\qual.dta";
string probename = "E:\\Documents\\Caltech\\CS156\\probe.dta";
string solname = "E:\\Documents\\Caltech\\CS156\\timeSVD\\solution";
string probesolname = "E:\\Documents\\Caltech\\CS156\\timeSVD\\probesolution";

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

		vector<int> curr(4);
		curr[0] = user - 1;
		curr[1] = movie - 1;
		curr[2] = rating;
		curr[3] = time; 

		ratings[index] = curr;

		index++;
	}

	file.close();

	probe = new int*[probe_size];

	ifstream probefile(probename);
	index = 0;

	while (getline(probefile, input))
	{
		sscanf(input.c_str(), "%d %d %d %d", &user, &movie, &time, &rating);

		probe[index] = new int[4];
		probe[index][0] = user - 1;
		probe[index][1] = movie - 1;
		probe[index][2] = rating;
		probe[index][3] = time; 

		index++;
	}

	probefile.close();

	ifstream qualfile(qualname);
	index = 0;

	while (getline(qualfile, input))
	{
		sscanf(input.c_str(), "%d %d %d", &user, &movie, &time);

		qual[index] = new int[3];
		qual[index][0] = user - 1;
		qual[index][1] = movie - 1;
		qual[index][2] = time;

		index++;
	}

	qualfile.close();
};

void initialize_svd()
{

	// Initialize all arrays with size userCount
	for (int i = 0; i < userCount; i++)
	{
		userIndices[i] = vector<int>(2);

		Bu[i] = 0.0;
		Alpha_u[i] = 0.0;
		Pu[i] = new double[k];
		for (int j = 0; j < k; j++)
			Pu[i][j] = 0.1 * (rand() / (RAND_MAX + 1.0)) / sqrt(k);
	}

	// Initialize all arrays with size movieCount
	for (int i = 0; i < movieCount; i++)
	{
		Bi[i] = 0.0;

		Yj[i] = new double[k];
		Qi[i] = new double[k];

		Bi_bin[i] = new double[bin_num];

		for (int j = 0; j < bin_num; j++)
			Bi_bin[i][j] = 0.0;

		for (int j = 0; j < k; j++)
		{
			Qi[i][j] = 0.1 * (rand() / (RAND_MAX + 1.0)) / sqrt(k);
			Yj[i][j] = 0.0;
		}
	}

	// Initilaize neighbors
	/*for (int i = 0; i < num_ratings; i++)
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
	
	for (int i = 0; i < probe_size; i++)
	{
		int user = probe[i][0];
		int movie = probe[i][1];

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
	qualfile.close();*/

	// Initialize user indices
	int currUser = 0; 
	int base = 0; 
	for (int i = 0; i < num_ratings; i++)
	{
		int user = ratings[i][0];

		if (user != currUser || i == num_ratings - 1)
		{
			userIndices[currUser][0] = base;
			userIndices[currUser][1] = i;
			currUser = user;
			base = i; 

			if (i == num_ratings - 1) userIndices[currUser][1] = i + 1;
		}
	}

	// Initialize neighbors 
	for (int i = 0; i < userCount; i++)
	{
		int start = userIndices[i][0];
		int end = userIndices[i][1];
		int count = end - start;

		map<int, double> tempMap; 

		vector<short> curr(count);

		double tavg = 0; 
		for (int j = start; j < end; j++)
		{
			curr[j - start] = ratings[j][1];
			tavg += ratings[j][3];

			tempMap[ratings[j][3]] = 0.000001; 
		}

		neighbors[i] = curr; 
		Tu[i] = tavg;
		Bu_t[i] = tempMap; 
	}

	// Add from qual 
	for (int i = 0; i < qual_size; i++)
	{
		int user = qual[i][0];
		int movie = qual[i][1];
		int time = qual[i][2];

		neighbors[user].push_back(movie);
		Tu[user] += time;
		Bu_t[user][time] = 0.000001;
	}

	// Add from probe 
	for (int i = 0; i < probe_size; i++)
	{
		int user = probe[i][0];
		int movie = probe[i][1];
		int time = probe[i][3];

		neighbors[user].push_back(movie);
		Tu[user] += time;
		Bu_t[user][time] = 0.000001;
	}

	// Define sizes and t_avg
	for (int i = 0; i < userCount; i++)
	{
		Tu[i] = Tu[i] / neighbors[i].size();
		//Nu[i] = pow(sqrt(neighbors[i].size()), -0.5);
		Nu[i] = pow(neighbors[i].size(), -0.5);
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

/*
int* getNeighbors(int user)
{
	int* implicit = new int[movieCount];

	for (int i = 0; i < movieCount; i++)
		implicit[i] = 0;

	for (int i = 0; i < maxRatings; i++)
		if (neighbors[user][i] == -1) break; 
		else implicit[neighbors[user][i]] = 1; 

	return implicit;
};*/

double* userVector(int user)
{
	double mag = Nu[user];

	double* sumYj = new double[k];

	for (int i = 0; i < k; i++)
		sumYj[i] = 0; 

	for (int i = 0; i < neighbors[user].size(); i++)
	{
		int curr = neighbors[user][i];
		addVectors(sumYj, Yj[curr], k);
	}
	
	multiplyVectorByScalar(sumYj, mag, k); 

	return sumYj; 
};

double dev_u(int user, int time)
{
	double sign = 1;
	double tu = Tu[user];
	if (time - tu < 0) sign = -1;
	double dev = sign * pow(abs(time - tu), 0.4);
	return dev;
};

int get_bin(int time)
{
	int bin_size = (max_time - 1) / bin_num;
	return time / bin_size;
};

double predict(int user, int movie, int time)
{
	double dev = dev_u(user, time);
	double bu = Bu[user] + Alpha_u[user] * dev + Bu_t[user][time];

	int bin = get_bin(time);
	double bi = Bi[movie] + Bi_bin[movie][bin];

	double* p = userVector(user);
	double* pu = Pu[user];
	addVectors(p, pu, k);

	double product = dot(p, Qi[movie], 0, k, 0);
	double result = avgRating + bu + bi + product;

	delete p;

	return result;
};

double probeRMSE()
{
	double sum = 0;

	for (int i = 0; i < probe_size; i++)
	{
		int user = probe[i][0];
		int movie = probe[i][1];
		int rating = probe[i][2];
		int time = probe[i][3];
		sum += pow(rating - predict(user, movie, time), 2);
	}

	double RMSE = sqrt(sum / probe_size);
	return RMSE;
};

void generate_qual(int num)
{
	string filename = solname + std::to_string(num) + ".dta";
	cout << filename << endl;
	ofstream solution(filename);

	for (int i = 0; i < qual_size; i++)
	{
		int user = qual[i][0];
		int movie = qual[i][1];
		int time = qual[i][2];
		double prediction = predict(user, movie, time);
		if (prediction > 5) prediction = 5;
		if (prediction < 1) prediction = 1;

		solution << prediction << std::endl;
	}

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
		int time = probe[i][3];
		double prediction = predict(user, movie, time);
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
	int user = 0;
	int base = 0; 
	auto begin = std::begin(ratings);

	for (int i = 0; i < num_ratings; i++)
	{
		if (ratings[i][0] != user)
		{
			std::shuffle(begin + base, begin + i - 1, g);
			base = i;
			user++;
		}
	}

	std::shuffle(std::begin(userIndices), std::end(userIndices), g);
};

void copyVector(double* a, double* b, int l)
{
	for (int i = 0; i < l; i++)
	{
		a[i] = b[i];
	}
};

void printVector(double* a, int l)
{
	for (int i = 0; i < l; i++)
		cout << a[i] << ", "; 
};

void train_svd()
{
	
	for (int e = 0; e < epochs; e++)
	{
		std::cout << "Starting Epoch : " << e << std::endl;
		
		chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
		shuffleRatings();

		int counter = 0;
		
		double errs = 0; 

		for(int kl = 0; kl < userCount; kl++)
		{
			int start = userIndices[kl][0];
			int end = userIndices[kl][1];

			int user = ratings[start][0];
			double mag = Nu[user];

			double* uvectorc = userVector(user);
			
			for (int i = start; i < end; i++)
			{
				counter += 1;
				int movie = ratings[i][1];
				int rating = ratings[i][2];
				int time = ratings[i][3];

				double* qi = Qi[movie];
				double* pu = Pu[user];
				double* uvector = new double[k];
				copyVector(uvector, uvectorc, k);
				addVectors(uvector, pu, k);

				double dev = dev_u(user, time);
				double bu = Bu[user] + Alpha_u[user] * dev + Bu_t[user][time];

				int bin = get_bin(time);
				double bi = Bi[movie] + Bi_bin[movie][bin];
				double product = dot(uvector, qi, 0, k, 0);
				
				double prediction = avgRating + bu + bi + product;

				double err = rating - prediction;
				errs += err * err; 

				Bu[user] += 0.001 * (err - 0.015 * bu);
				Bi[movie] += 0.001 * (err - 0.015 * bi);
				Alpha_u[user] += 0.00001 * (err * dev - 0.0004 * Alpha_u[user]);
				Bu_t[user][time] += 0.001 * (err - 0.015 * Bu_t[user][time]);
				Bi_bin[movie][bin] += 0.001 * (err - 0.015 * Bi_bin[movie][bin]);

				if (i == start)
				{
					for (int j = 0; j < neighbors[user].size(); j++)
					{
						int index = neighbors[user][j];

						for (int z = 0; z < k; z++)
							Yj[index][z] += 0.001 * (err * mag * qi[z] - 0.015 * Yj[index][z]);
					}

					delete uvectorc;
					uvectorc = userVector(user);
				}

				for (int j = 0; j < k; j++)
				{
					pu[j] += 0.005 * (err * qi[j] - 0.015 * pu[j]);
					qi[j] += 0.005 * (err * uvector[j] - 0.015 * qi[j]);
				}

				if (i % 10000000 == 0)
				{
					/*std::cout << "Pass:  " << i;
					std::cout << ", Prediction:  " << prediction;
					std::cout << ", Err:  " << err << std::endl;*/
				}

				delete uvector;
			}

			delete uvectorc; 
		}

		std::cout << "FINISHED EPOCH:  " << e << std::endl;
		std::cout << "Train RMSE:  " << sqrt(errs / num_ratings)  << std::endl;
		std::cout << "Probe RMSE:  " << probeRMSE() << std::endl;

		chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
		auto duration = chrono::duration_cast<chrono::seconds>(t2 - t1).count();
		cout << "Time elapsed: " << duration << endl;

		//reg = reg * 0.95;
		//reg2 = reg2 * 0.9;

		if (e % 4 == 0 && e > 3 || e == 29 || e == 30)
		{
			generate_qual(e);
			generate_probesols(e);
		}
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
