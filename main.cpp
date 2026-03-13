#include <opencv2/opencv.hpp>
#include <vector>

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) return -1;

    cv::Mat frame, gray, blurred, edges;

    while (true) {
    cap >> frame;
    if (frame.empty()) break;

    cv::Mat gray, thresh, edges;
    
    // 1. Convertire și Threshold (transformăm tot ce e alb în alb pur, restul negru)
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);
    
    // Folosim Threshold în loc de Canny pentru că fundalul e luminos
    cv::threshold(gray, thresh, 120, 255, cv::THRESH_BINARY);

    // 2. Găsirea contururilor
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        
        // Dacă forma e destul de mare (ajustăm la 3000 pentru a fi mai sensibil)
        if (area > 3000) {
            double peri = cv::arcLength(contour, true);
            std::vector<cv::Point> approx;
            cv::approxPolyDP(contour, approx, 0.02 * peri, true);

            // Desenăm orice formă cu 4-5 colțuri (uneori degetul adaugă un colț)
            if (approx.size() >= 4 && approx.size() <= 6) {
                cv::drawContours(frame, std::vector<std::vector<cv::Point>>{approx}, -1, cv::Scalar(0, 255, 0), 4);
                cv::putText(frame, "CARTE", approx[0], cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
            }
        }
    }

    // IMPORTANT: Afișăm și imaginea procesată ca să vedem unde e problema
    cv::imshow("Ce vede algoritmul (Thresh)", thresh);
    cv::imshow("Blackjack Vision", frame);

    if (cv::waitKey(30) >= 0) break;
    }
    return 0;
}