struct InputParams {
    std::string model_binary_path;
    int iterations;
    std::string img_path;
    std::string label_file_path;
};

enum class InputType {
    Classifier,
    Detector
};

bool ParseInputs(int argc, char* argv[], InputType input_type, InputParams& params) {
    if (input_type == InputType::Classifier && argc != 5) {
        std::cerr << "Invalid number of arguments. Usage: program_name model_binary_path iterations img_path label_file_path\n";
        return false;
    }
    else if (input_type == InputType::Detector && argc != 4) {
        std::cerr << "Invalid number of arguments. Usage: program_name model_binary_path iterations img_path\n";
        return false;
    }

    params.model_binary_path = argv[1];
    params.img_path = argv[3];

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

    // Check if img_path file exists
    std::ifstream img_file(params.img_path);
    if (!img_file) {
        std::cerr << "Image file does not exist: " << params.img_path << std::endl;
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

    return true;
}