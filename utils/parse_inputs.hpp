struct InputParams {
    std::string model_binary_path;
    int iterations;
    std::string input_image_path;
    std::string label_file_path;
    std::string model_name;
};

enum class InputType {
    Classifier,
    Detector
};

bool ParseInputs(int argc, char* argv[], InputType input_type, InputParams& params) {
    if (input_type == InputType::Classifier && argc != 5) {
        std::cerr << "Invalid number of arguments. Usage: program_name model_binary_path iterations input_image_path label_file_path\n";
        return false;
    }
    else if (input_type == InputType::Detector && argc != 5) {
        std::cerr << "Invalid number of arguments. Usage: program_name model_binary_path iterations input_image_path model_type\n";
        return false;
    }

    params.model_binary_path = argv[1];
    params.input_image_path = argv[3];

    try {
        params.iterations = std::stoi(argv[2]);
    } catch (const std::exception& e) {
        std::cerr << "Invalid iterations argument. It must be an integer.\n";
        return false;
    }

    // Check if model_binary_path file exists
    std::ifstream model_file(params.model_binary_path);
    if (!model_file) {
        std::cerr << "Model binary file does not exist: " << params.model_binary_path << std::endl;
        return false;
    }

    // Check if input_image_path file exists
    std::ifstream img_file(params.input_image_path);
    if (!img_file) {
        std::cerr << "Image file does not exist: " << params.input_image_path << std::endl;
        return false;
    }

    if (input_type == InputType::Classifier) {
        params.label_file_path = argv[4];

        // Check if label_name file exists
        std::ifstream label_file(params.label_file_path);
        if (!label_file) {
            std::cerr << "Label file does not exist: " << params.label_file_path << std::endl;
            return false;
        }
    }
    else if(input_type == InputType::Detector) {

        std::vector<std::string> supported_models = {"YOLO", "NANODET", "EFFICIENTDET","MOBNETSSD"};
        
        params.model_family = argv[4];
        if (std::find(supported_models.begin(), supported_models.end(), params.model_family) == supported_models.end()) {
        // Element not found
            std::cerr << "Invalid model type, supported: YOLO, MOBNETSSD, EFFICIENTDET, NANODET\n";
            return false;
        }
    }

    return true;
}

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <regex>
#include <stdexcept>
#include <unordered_map>

std::unordered_map<char, int> getLayoutDims(const std::string& layout, const std::string& shape) {
    std::unordered_map<char, int> layoutDims;

    std::regex layoutPattern(R"(\w+)"); // Regular expression to find letters in the layout string
    std::smatch layoutMatches;
    
    // Regular expression to find tuples in the shape string
    std::regex shapePattern(R"(\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\))");
    std::smatch shapeMatches;

    if (std::regex_search(layout, layoutMatches, layoutPattern) &&
        std::regex_search(shape, shapeMatches, shapePattern)) {

        std::string lettersStr = layoutMatches.str(0); // Extract the matched string
        std::vector<char> letters;

        // Extract characters from the string and store them in a vector
        for (char letter : lettersStr) {
            if (std::isalpha(letter)) { // Check if the character is a letter
                letters.push_back(letter);
            }
        }

        std::vector<int> numbers;

        // Extract numbers from the shape string
        for (size_t i = 1; i < shapeMatches.size(); ++i) {
            numbers.push_back(std::stoi(shapeMatches.str(i)));
        }

        if (letters.size() != numbers.size()) {
            throw std::invalid_argument("Layout and shape sizes do not match.");
        }

        // Create the layout dimensions mapping
        for (size_t i = 0; i < letters.size(); ++i) {
            layoutDims[letters[i]] = numbers[i];
        }
    } else {
        throw std::invalid_argument("Failed to extract layout and shape values.");
    }

    return layoutDims;
}

