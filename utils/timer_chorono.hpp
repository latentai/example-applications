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
            return std::accumulate(elapsed_times.begin(), elapsed_times.end(), 0.0) / elapsed_times.size();
    }
};