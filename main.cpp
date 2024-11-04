#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "alg.cpp"

using namespace std;
using namespace std::chrono;

void print_help() {
    cout << "--- Help ---" << endl;
    cout << "tsp_manual - solve the TSP problem with a manually entered distance matrix" << endl;
    cout << "tsp_random - solve the TSP problem with a randomly generated distance matrix" << endl;
    cout << "tsp_file - solve the TSP problem with a distance matrix from a file" << endl;
    cout << "generate - generate a TSP instance file with random coordinates" << endl;
    cout << "generate_fast - generate a TSP instance file with random coordinates (default range)" << endl;
    cout << "help - show this help message" << endl;
    cout << "exit - exit the program" << endl;
}

string get_command() {
    string command;
    cout << "command> ";
    cin >> command;
    return command;
}

int main() {
    while (true) {
        string command = get_command();
        if (command == "exit") {
            break;
        } else if (command == "tsp_manual") {
            vector<vector<int>> dist_matrix = input_matrix();
            cout << "Input matrix:" << endl;
            for (const auto& row : dist_matrix) {
                for (int val : row) {
                    cout << val << " ";
                }
                cout << endl;
            }
            auto start = high_resolution_clock::now();
            auto [path, distance] = greedy_tsp(dist_matrix);
            auto end = high_resolution_clock::now();
            cout << "Shortest path: ";
            for (int city : path) {
                cout << city << " ";
            }
            cout << endl;
            cout << "Total distance: " << distance << endl;
            cout << "Execution time: " << duration_cast<microseconds>(end - start).count() << " microseconds" << endl;
        } else if (command == "tsp_random") {
            int n;
            cout << "Enter the size of the matrix: ";
            cin >> n;
            vector<vector<int>> dist_matrix = generate_random_dist_matrix(n);
            cout << "Randomly generated matrix:" << endl;
            for (const auto& row : dist_matrix) {
                for (int val : row) {
                    cout << val << " ";
                }
                cout << endl;
            }
            auto start = high_resolution_clock::now();
            auto [path, distance] = greedy_tsp(dist_matrix);
            auto end = high_resolution_clock::now();
            cout << "Shortest path: ";
            for (int city : path) {
                cout << city << " ";
            }
            cout << endl;
            cout << "Total distance: " << distance << endl;
            cout << "Execution time: " << duration_cast<microseconds>(end - start).count() << " microseconds" << endl;
        } else if (command == "tsp_file") {
            string file_name;
            cout << "Enter file name: ";
            cin >> file_name;
            vector<vector<int>> dist_matrix = matrix_file(file_name);
            cout << "Input matrix:" << endl;
            for (const auto& row : dist_matrix) {
                for (int val : row) {
                    cout << val << " ";
                }
                cout << endl;
            }
            auto start = high_resolution_clock::now();
            auto [path, distance] = greedy_tsp(dist_matrix);
            auto end = high_resolution_clock::now();
            cout << "Shortest path: ";
            for (int city : path) {
                cout << city << " ";
            }
            cout << endl;
            cout << "Total distance: " << distance << endl;
            cout << "Execution time: " << duration_cast<microseconds>(end - start).count() << " microseconds" << endl;
        } else if (command == "generate") {
            int num_cities;
            cout << "Enter number of cities: ";
            cin >> num_cities;
            string file_name;
            cout << "Enter file name: ";
            cin >> file_name;
            pair<int, int> range_x, range_y;
            cout << "Enter range for x coordinates (min, max): ";
            cin >> range_x.first >> range_x.second;
            cout << "Enter range for y coordinates (min, max): ";
            cin >> range_y.first >> range_y.second;
            generate_file(num_cities, file_name, range_x, range_y);
        } else if (command == "generate_fast") {
            int num_cities;
            cout << "Enter number of cities: ";
            cin >> num_cities;
            string file_name;
            cout << "Enter file name: ";
            cin >> file_name;
            generate_file(num_cities, file_name);
        } else if (command == "help") {
            print_help();
        } else {
            cout << "Invalid command. Type \"help\" for a list of commands." << endl;
        }
    }
    return 0;
}