#include <chrono>
#include <vector>
#include <numeric>

class Timer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::vector<double> elapsed_times;
    double ms_factor{0.000001};


public:
    Timer() {}

    void start() {
            start_time = std::chrono::high_resolution_clock::now();        
    }

    void stop() {
            std::chrono::time_point<std::chrono::high_resolution_clock> end_time = std::chrono::high_resolution_clock::now();
            auto elapsed_ms = (end_time - start_time).count() * ms_factor;
            elapsed_times.push_back(elapsed_ms);
    }

    double averageElapsedMilliseconds() const {
        if (elapsed_times.empty())
            return 0.0;
        else
            return std::accumulate(elapsed_times.begin() + 1, elapsed_times.end(), 0.0) / (elapsed_times.size() - 1);
    }

    double standardDeviationMilliseconds() const {
        if (elapsed_times.size() < 2)
            return 0.0;

        double mean = averageElapsedMilliseconds();
        double accum = 0.0;
        std::for_each(elapsed_times.begin() + 1, elapsed_times.end(), [&](const double d) {
            accum += (d - mean) * (d - mean);
        });

        return std::sqrt(accum / (elapsed_times.size() - 2));
    }

};

// Function to round a double to a specified number of decimal places
double roundToDecimalPlaces(double value, int decimal_places) {
    double factor = std::pow(10.0, decimal_places);
    return std::round(value * factor) / factor;
}