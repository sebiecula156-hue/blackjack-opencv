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
        cap >> frame; // Captează un cadru nou
        if (frame.empty()) break;

        cv::imshow("Test Blackjack Camera", frame); // Afișează imaginea

        if (cv::waitKey(30) >= 0) break;
    }

    return 0;
}