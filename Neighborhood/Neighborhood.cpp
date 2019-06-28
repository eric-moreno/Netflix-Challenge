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
int epochs = 17;

double reg = 0.005;
double reg2 = 0.005;

const int num_ratings = 98291669;
vector<vector <int> > ratings(num_ratings);

double* Bi = new double[movieCount];
double* Bu = new double[userCount];

double* Buj_user = new double[userCount];
double* Buj_item = new double[movieCount];

// Pair of movie, rating 
vector<vector < pair<short, short> > > Ru(userCount);
vector<vector < short > > Nu(userCount);

double** Wij = new double*[movieCount];
double** Cij = new double*[movieCount];

vector<vector <int> > userIndices(userCount);

int probe_size = 1374739;
int** probe;

int qual_size = 2749898;
int** qual = new int*[qual_size]; 

string basename = "E:\\Documents\\Caltech\\CS156\\umbase.dta";
string qualname = "E:\\Documents\\Caltech\\CS156\\qual.dta";
string probename = "E:\\Documents\\Caltech\\CS156\\probe.dta";
string solname = "E:\\Documents\\Caltech\\CS156\\NGB\\solution";
string probesolname = "E:\\Documents\\Caltech\\CS156\\NGB\\probesolution";
string baseMUname = "E:\\Documents\\Caltech\\CS156\\base.dta";

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
		Buj_user[i] = 0.0;
	}

	// Initialize all arrays with size movieCount
	for (int i = 0; i < movieCount; i++)
	{
		Bi[i] = 0.0;
		Buj_item[i] = 0.0; 
		Wij[i] = new double[movieCount];
		Cij[i] = new double[movieCount];

		for (int j = 0; j < movieCount; j++)
		{
			Wij[i][j] = 0.0;//0.1 * (rand() / (RAND_MAX + 1.0)) / sqrt(k);
			Cij[i][j] = 0.0;//0.1 * (rand() / (RAND_MAX + 1.0)) / sqrt(k);
		}
	}

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

	// Initialize R(u)
	for (int i = 0; i < userCount; i++)
	{
		int start = userIndices[i][0];
		int end = userIndices[i][1];
		int count = end - start;

		vector< pair<short, short> > curr(count);

		for (int j = start; j < end; j++)
		{
			pair <short, short> p = make_pair(ratings[j][1], ratings[j][2]);
			curr[j - start] = p;
		}

		Ru[i] = curr;
	}
	

	// Initialize N(u)
	for (int i = 0; i < userCount; i++)
	{
		int s = Ru[i].size();
		vector< short > curr(s);

		for (int j = 0; j < s; j++)
			curr[j] = Ru[i][j].first;

		Nu[i] = curr; 
	}

	for (int i = 0; i < probe_size; i++)
	{
		int user = probe[i][0];
		int movie = probe[i][1];
		Nu[user].push_back(movie);
	}

	for (int i = 0; i < qual_size; i++)
	{
		int user = qual[i][0];
		int movie = qual[i][1];
		Nu[user].push_back(movie);
	}

};

void Buj_init()
{
	/*for (int e = 0; e < 30; e++)
	{
		for(int i = 0; i < )
		for (int i = 0; i < num_ratings; i++)
		{
			int user = ratings[i][0];
			int movie = ratings[i][1];
			int rating = ratings[i][2];

			double prediction = avgRating + Buj_user[user] + Buj_item[movie];
			double err = rating - prediction;
			Buj_user[user] += 0.002 * (err - 0.005 * Buj_user[user]);
			Buj_item[movie] += 0.002 * (err - 0.005 * Buj_item[movie]);

			//Bu[user] += 0.002 * (err - 0.005 * Buj_user[user]);
			//Bi[movie] += 0.002 * (err - 0.005 * Buj_item[movie]);
		}
	}*/

	/*for (int i = 0; i < movieCount; i++)
	{
		vector<short> curr; 
		for (int j = 0; j < num_ratings; j++)
		{
			if (ratings[j][1] == i)
			{
				curr.push_back(ratings[j][2]);
			}
		}

		double sum = 0;
		for (int j = 0; j < curr.size(); j++)
			sum += curr[j] - avgRating;

		Buj_item[i] = sum / (25 + sqrt(curr.size()));
	}*/

	ifstream file(baseMUname);
	string input;
	int user, movie, time, rating;
	int count = 0; 
	int currMovie = 0; 
	int index = 0; 
	double sum = 0; 

	while (getline(file, input))
	{
		sscanf(input.c_str(), "%d %d %d %d", &user, &movie, &time, &rating);

		if (movie != currMovie || index == num_ratings - 1)
		{
			Buj_item[currMovie] = sum / (25 + count);

			sum = 0;
			count = 0; 
			currMovie += 1; 
		}
		sum += rating - avgRating; 
		count += 1; 
		index += 1; 
	}

	file.close();


	for (int i = 0; i < userCount; i++)
	{
		double sum = 0;
		for (int j = 0; j < Ru[i].size(); j++)
			sum += (Ru[i][j].second - avgRating - Buj_item[Ru[i][j].first]);
		Buj_user[i] = sum / (10 + Ru[i].size());
	}
};

double Buj(int user, int movie)
{
	return avgRating + Buj_user[user] + Buj_item[movie];
}

double predict(int user, int movie, int time)
{
	double s1 = Ru[user].size();
	double sum1 = 0;
	for (int i = 0; i < s1; i++)
	{
		int j = Ru[user][i].first;
		int r = Ru[user][i].second;
		sum1 += (r - Buj(user, j)) * Wij[movie][j];
	}
	//sum1 = sum1 * pow(sqrt(s1), -0.5);
	sum1 = sum1 * pow(s1, -0.5);

	double s2 = Nu[user].size();
	double sum2 = 0;
	for (int i = 0; i < s2; i++)
	{
		int j = Nu[user][i];
		sum2 += Cij[movie][j];
	}
	//sum2 = sum2 * pow(sqrt(s2), -0.5);
	sum2 = sum2 * pow(s2, -0.5);

	double result = avgRating + Bu[user] + Bi[movie] + sum1 + sum2; 

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

		double prediction = predict(user, movie, time);
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
	std::shuffle(std::begin(ratings), std::end(ratings), g);
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

		double errs = 0; 

		for (int i = 0; i < num_ratings; i++)
		{
			int user = ratings[i][0];
			int movie = ratings[i][1];
			int rating = ratings[i][2];
			int time = ratings[i][3];

			double prediction = predict(user, movie, time);
			double err = rating - prediction; 

			double s1 = Ru[user].size();
			//double rs1 = pow(sqrt(s1), -0.5);
			double rs1 = pow(s1, -0.5);
			for (int j = 0; j < s1; j++)
			{
				int mov = Ru[user][j].first;
				int rmov = Ru[user][j].second;
				double val = 0.005 * (rs1 * err * (rmov - Buj(user, mov)) - 0.002 * Wij[movie][mov]);

				Wij[movie][mov] += val;
				Wij[mov][movie] += val;
			}

			double s2 = Nu[user].size(); 
			//double rs2 = pow(sqrt(s2), -0.5);
			double rs2 = pow(s2, -0.5);
			for (int j = 0; j < s2; j++)
			{
				int mov = Nu[user][j];
				double val = 0.005 * (rs2 * err - 0.002 * Cij[movie][mov]);
				Cij[movie][mov] += val;
				Cij[mov][movie] += val; 
			}

			Bu[user] += 0.005 * (err - 0.002 * Bu[user]);
			Bi[movie] += 0.005 * (err - 0.002 * Bi[movie]);

			if (err < -5 || err > 5)
					std::cout << "WARNING! " << i << "!" << endl; 

			if (i % 10000000 == 0)
			{
				std::cout << "Pass:  " << i;
				std::cout << ", Prediction:  " << prediction;
				std::cout << ", Err:  " << err << std::endl;
			}

			errs += err * err; 
		}

		cout << endl; 
		std::cout << "FINISHED EPOCH:  " << e << std::endl;
		std::cout << "Train RMSE:  " << sqrt(errs / num_ratings) << std::endl;
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
	Buj_init();
	std::cout << "Initialized neighborhood model..." << std::endl;

	std::cout << "Training model... " << std::endl;
	train_svd();

	std::cout << "Done. ";

	std::cin.get();
	return 0;
}
