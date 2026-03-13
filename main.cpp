#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Încearcă să deschidă camera (0 este de obicei camera web implicită)
    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cout << "Eroare: Nu pot accesa camera!" << std::endl;
        return -1;
    }

    std::cout << "OpenCV functioneaza! Apasa orice tasta in fereastra video pentru a inchide." << std::endl;

    cv::Mat frame;
    while (true) {
    cap >> frame;
    if (frame.empty()) break;

    cv::Mat gray, blurred, edges;
    
    // 1. Convertire în alb-negru
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    
    // 2. Blur pentru a ignora detaliile inutile
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    
    // 3. Detectare margini
    cv::Canny(blurred, edges, 75, 200);

    // Afișăm marginile detectate
    cv::imshow("Detectie Margini", edges);
    cv::imshow("Camera Normala", frame);

    if (cv::waitKey(30) >= 0) break;
    }

    return 0;
}