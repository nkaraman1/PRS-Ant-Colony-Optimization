#include <iostream>
#include <fstream>
#include <random>

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Invalid arguments!\n";
        exit(EXIT_FAILURE);
    }

    int number_of_nodes = std::stoi(argv[1]);
    std::random_device rd;
    std::mt19937 gen(rd());

    std::ofstream map_file("map.txt");
    for (int i = 0; i < number_of_nodes; i++) {
        for (int j = 0; j < number_of_nodes; j++) {
            double weight = 0.0;
            if (i != j) {
                std::uniform_real_distribution<double> distribution(0.0, 50.0);
                weight = distribution(gen);
            }
            map_file << i << " " << j << " " << weight << "\n";
        }
    }

    map_file.close();
    return 0;
}