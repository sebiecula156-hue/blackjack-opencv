#include <opencv2/opencv.hpp>
#include <vector>

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) return -1;

    cv::Mat frame, gray, blurred, edges;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // 1. Pregătirea imaginii
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
        cv::Canny(blurred, edges, 75, 200);

        // 2. Găsirea contururilor
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours) {
            // Calculăm perimetrul pentru a aproxima forma
            double peri = cv::arcLength(contour, true);
            std::vector<cv::Point> approx;
            cv::approxPolyDP(contour, approx, 0.02 * peri, true);

            // Dacă forma are 4 colțuri și o mărime minimă, e probabil o carte
            if (approx.size() == 4 && cv::contourArea(contour) > 5000) {
                // Desenăm conturul verde peste imaginea originală
                cv::drawContours(frame, std::vector<std::vector<cv::Point>>{approx}, -1, cv::Scalar(0, 255, 0), 3);
                
                // Punem un text deasupra
                cv::putText(frame, "Carte detectata", approx[0], cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
            }
        }

        cv::imshow("Blackjack Vision", frame);
        if (cv::waitKey(30) >= 0) break;
    }
    return 0;
}