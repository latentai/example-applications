#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <regex>
#include <stdexcept>
#include <unordered_map>
#include <stdexcept>


struct InputParams {
    std::string model_binary_path;
    int iterations;
    std::string input_image_path;
    std::string label_file_path;
    std::string model_family;
};

enum class InputType {
    Classifier,
    Detector
};

std::vector<std::string> supported_models = {"YOLO", "NANODET", "EFFICIENTDET","MOBNETSSD"};
std::vector<std::string> supported_precisions = {"float32","float16","int8"};

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
        
        params.model_family = argv[4];
        if (std::find(supported_models.begin(), supported_models.end(), params.model_family) == supported_models.end()) {
        // Element not found
            std::cerr << "Invalid model type, supported: YOLO, MOBNETSSD, EFFICIENTDET, NANODET\n";
            return false;
        }
    }

    return true;
}

bool validateClassifierArguments(std::map<std::string, std::string>& cmdArgs) {
    const std::vector<std::string> requiredArgs = {"--model_path", "--img_path", "--iterations", "--precision","--label_file"};
    
    for (const auto& arg : requiredArgs) {
        auto it = cmdArgs.find(arg);
        if (it == cmdArgs.end() || it->second.empty()) {
            throw std::runtime_error("Required argument missing or empty: " + arg);
        }
    }

    // Check if model_binary_path file exists
    std::ifstream model_file(cmdArgs["--model_path"]);
    if (!model_file) {
        throw std::runtime_error("Model binary file does not exist: " + cmdArgs["--model_path"]);
    }

    // Check if input_image_path file exists
    std::ifstream img_file(cmdArgs["--img_path"]);
    if (!img_file) {
        throw std::runtime_error("Image file does not exist: " + cmdArgs["--img_path"]);
    }

    // Check if input_image_path file exists
    std::ifstream label_file(cmdArgs["--label_file"]);
    if (!label_file) {
        throw std::runtime_error("label file does not exist: " + cmdArgs["--label_file"]);
    }


    if (std::find(supported_precisions.begin(), supported_precisions.end(),cmdArgs["--precision"]) == supported_precisions.end()) {
       throw std::runtime_error("Supported precisions: float32, float16 and int8 \n");
    }

    try {
        std::stoi(cmdArgs["--iterations"]);
    } catch (const std::exception& e) {
       throw std::runtime_error("Invalid iterations argument. It must be an integer.\n");
    }

    return true;
}

bool validateArguments(std::map<std::string, std::string>& cmdArgs) {
    const std::vector<std::string> requiredArgs = {"--model_path", "--img_path", "--iterations", "--model_family",
    "--iou_thres","--conf_thres","--precision"};
    
    for (const auto& arg : requiredArgs) {
        auto it = cmdArgs.find(arg);
        if (it == cmdArgs.end() || it->second.empty()) {
            throw std::runtime_error("Required argument missing or empty: " + arg);
        }
    }

    // Check if model_binary_path file exists
    std::ifstream model_file(cmdArgs["--model_path"]);
    if (!model_file) {
        throw std::runtime_error("Model binary file does not exist: " + cmdArgs["--model_path"]);
    }

    // Check if input_image_path file exists
    std::ifstream img_file(cmdArgs["--img_path"]);
    if (!img_file) {
        throw std::runtime_error("Image file does not exist: " + cmdArgs["--img_path"]);
    }


    if (std::find(supported_models.begin(), supported_models.end(),cmdArgs["--model_family"]) == supported_models.end()) {
    // Element not found
       throw std::runtime_error("Invalid model type, supported: YOLO, MOBNETSSD, EFFICIENTDET, NANODET\n");
    }

    if (std::find(supported_precisions.begin(), supported_precisions.end(),cmdArgs["--precision"]) == supported_precisions.end()) {
       throw std::runtime_error("Supported precisions: float32, float16 and int8 \n");
    }

    try {
        std::stoi(cmdArgs["--iterations"]);
    } catch (const std::exception& e) {
       throw std::runtime_error("Invalid iterations argument. It must be an integer.\n");
    }

    try {
        std::stof(cmdArgs["--conf_thres"]);
    } catch (const std::exception& e) {
        throw std::runtime_error("Confidence threshold must be a float.");
    }

    try {
        std::stof(cmdArgs["--iou_thres"]);
    } catch (const std::exception& e) {
        throw std::runtime_error("IOU threshold must be a float.");
    }


    return true;
}

// Function to parse input arguments
std::map<std::string, std::string> parseInput(int argc, char* argv[]) {
    std::map<std::string, std::string> cmdArgs;

    for (int i = 1; i < argc; i += 2) {
        if (i + 1 < argc) { // Make sure we have a pair
            std::string key(argv[i]);
            if (key[0] == '-' && key[1] == '-') {
                cmdArgs[key] = argv[i + 1];
            } else {
                std::cerr << "Invalid argument format: " << key << std::endl;
                throw std::invalid_argument("Invalid argument format \n Usage program_name --model_path </path/to/modelLibrary.so> --img_path </path/to/imput image> --iterations <number of iterations> --model_family <model family> --iou_thres <iou threshold> --conf_thres <confidence threshold> --precision <precision>");
            }
        } else {
            std::cerr << "Missing value for argument: " << argv[i] << std::endl;
            throw std::invalid_argument("Missing argument value");
        }
    }

    validateArguments(cmdArgs);

    return cmdArgs;
}

std::map<std::string, std::string> parseClassifierInput(int argc, char* argv[]) {
    std::map<std::string, std::string> cmdArgs;

    for (int i = 1; i < argc; i += 2) {
        if (i + 1 < argc) { // Make sure we have a pair
            std::string key(argv[i]);
            if (key[0] == '-' && key[1] == '-') {
                cmdArgs[key] = argv[i + 1];
            } else {
                std::cerr << "Invalid argument format: " << key << std::endl;
                throw std::invalid_argument("Invalid argument format \n Usage program_name --model_path </path/to/modelLibrary.so> --img_path </path/to/imput image> --iterations <number of iterations> --label_file <path/to/labelsfile> --precision <precision>");
            }
        } else {
            std::cerr << "Missing value for argument: " << argv[i] << std::endl;
            throw std::invalid_argument("Missing argument value");
        }
    }

    validateClassifierArguments(cmdArgs);

    return cmdArgs;
}

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

