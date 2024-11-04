#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <random>
#include <chrono>
#include <limits>

using namespace std;

pair<vector<int>, int> greedy_tsp(const vector<vector<int>>& dist_matrix) {
    int n = dist_matrix.size();
    vector<bool> visited(n, false);
    vector<int> path;
    
    int current_city = 0;
    path.push_back(current_city);
    visited[current_city] = true;
    int total_distance = 0;
    
    for (int i = 0; i < n - 1; ++i) {
        int nearest_city = -1;
        int min_distance = numeric_limits<int>::max();
        
        // Find the nearest unvisited city
        for (int city = 0; city < n; ++city) {
            if (!visited[city] && dist_matrix[current_city][city] < min_distance) {
                nearest_city = city;
                min_distance = dist_matrix[current_city][city];
            }
        }
        
        // Move to the nearest city
        path.push_back(nearest_city);
        visited[nearest_city] = true;
        total_distance += min_distance;
        current_city = nearest_city;
    }
    
    // Return to the start city
    total_distance += dist_matrix[current_city][path[0]];
    path.push_back(path[0]);
    
    return {path, total_distance};
}

vector<vector<int>> input_matrix() {
    int n;
    cout << "Enter the size of the matrix: ";
    cin >> n;
    vector<vector<int>> matrix(n, vector<int>(n));
    
    cout << "Enter the matrix row by row (space-separated values):" << endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cin >> matrix[i][j];
        }
    }
    
    return matrix;
}

vector<vector<int>> generate_random_dist_matrix(int n) {
    vector<vector<int>> matrix(n, vector<int>(n));
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(1, 100);
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                matrix[i][j] = dis(gen);
            } else {
                matrix[i][j] = 0;
            }
        }
    }
    
    return matrix;
}

vector<vector<int>> matrix_file(const string& filename) {
    vector<pair<int, int>> coordinates;
    ifstream file(filename);
    int num_cities;
    file >> num_cities;
    
    for (int i = 0; i < num_cities; ++i) {
        int index, x, y;
        file >> index >> x >> y;
        coordinates.emplace_back(x, y);
    }
    
    vector<vector<int>> dist_matrix(num_cities, vector<int>(num_cities));
    for (int i = 0; i < num_cities; ++i) {
        for (int j = 0; j < num_cities; ++j) {
            int dx = coordinates[i].first - coordinates[j].first;
            int dy = coordinates[i].second - coordinates[j].second;
            dist_matrix[i][j] = round(sqrt(dx * dx + dy * dy));
        }
    }
    
    return dist_matrix;
}

void generate_file(int num_cities, const string& filename = "tsp_instance.txt", pair<int, int> x_range = {0, 2000}, pair<int, int> y_range = {0, 2000}) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis_x(x_range.first, x_range.second);
    uniform_int_distribution<> dis_y(y_range.first, y_range.second);
    
    ofstream file(filename);
    file << num_cities << endl;
    for (int i = 0; i < num_cities; ++i) {
        int x = dis_x(gen);
        int y = dis_y(gen);
        file << i + 1 << " " << x << " " << y << endl;
    }
    
    cout << "Generated TSP file with " << num_cities << " cities: " << filename << endl;
}