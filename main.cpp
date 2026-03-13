#include <opencv2/opencv.hpp>
#include <vector>

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) return -1;

    cv::Mat frame, gray, blurred, edges;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat gray, thresh;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

        // Folosim Adaptive Threshold pentru a compensa umbrele și lumina peretelui
        cv::adaptiveThreshold(gray, thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);
        cv::bitwise_not(thresh, thresh); // Invertim pentru a avea obiectul alb pe fundal negru

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            
            // 1. Filtru de mărime: ignorăm ce e prea mic sau prea gigant (ca peretele)
            if (area > 5000 && area < (frame.cols * frame.rows * 0.5)) {
                double peri = cv::arcLength(contour, true);
                std::vector<cv::Point> approx;
                cv::approxPolyDP(contour, approx, 0.02 * peri, true);

                if (approx.size() == 4) {
                    // 2. Filtru de Raport de Aspect
                    cv::Rect rect = cv::boundingRect(approx);
                    float ar = (float)rect.width / (float)rect.height;
                    
                    // O carte de joc are raportul de aprox 0.6 - 0.8 (sau invers daca e rotita)
                    if ((ar > 0.5 && ar < 0.9) || (ar > 1.1 && ar < 1.7)) {
                        cv::drawContours(frame, std::vector<std::vector<cv::Point>>{approx}, -1, cv::Scalar(0, 255, 0), 4);
                        cv::putText(frame, "CARTE OK", approx[0], cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
                    }
                }
            }
        }

        cv::imshow("Dupa Filtrare (Thresh)", thresh);
        cv::imshow("Blackjack Vision", frame);

        if (cv::waitKey(30) >= 0) break;
    }
    return 0;
}