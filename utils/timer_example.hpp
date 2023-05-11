#include <chrono>
#include <iostream>
#include <string>

class Timer {
public:
    Timer(const std::string& reference = "", bool printDuration = true)
        : m_PrintDuration(printDuration),
          m_Reference(reference),
          m_StartTimePoint(std::chrono::high_resolution_clock::now()) {}

    ~Timer() {
        Stop();
    }

    void Stop() {
        auto endTimePoint = std::chrono::high_resolution_clock::now();
        m_Duration = (endTimePoint - m_StartTimePoint).count() * ms_factor;

        if (m_PrintDuration && !m_DurationPrinted) {
            std::cout << m_Reference << " Time: " << m_Duration << "ms\n";
            m_DurationPrinted = true;
        }
    }

    double GetDuration() const {
        return m_Duration;
    }


private:
    double ms_factor{0.000001};
    bool m_PrintDuration, m_DurationPrinted{false};
    std::string m_Reference;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimePoint;
    double m_Duration;
};

struct TimeOperations {
  std::vector<Timer> preprocessing;
  std::vector<Timer> inference;
  std::vector<Timer> postprocessing;
};


inline double CalculateAverageTime(const std::vector<Timer>& timers) {
    double totalDuration = 0.0;
    for (const auto& timer : timers) {
        totalDuration += timer.GetDuration();
    }
    double averageDuration = totalDuration / timers.size();
    return averageDuration;
};

inline void PrintOperationsStats(const TimeOperations& op_timer,int iterations){
    auto elements_to_average{iterations};
    std::cout << "-------------- Timing after " << iterations + 1 << " iterations --------------------" << "\n";
    std::cout << "Average Preprocessing Time  : " << CalculateAverageTime(std::vector<Timer>((op_timer.preprocessing.end()-elements_to_average), op_timer.preprocessing.end())) << " ms\n";
    std::cout << "Average Inference Time      : " << CalculateAverageTime(std::vector<Timer>((op_timer.inference.end()-elements_to_average), op_timer.inference.end())) << " ms\n";
    std::cout << "Average Postprocessing Time : " << CalculateAverageTime(std::vector<Timer>((op_timer.postprocessing.end()-elements_to_average), op_timer.postprocessing.end())) << " ms\n";

};