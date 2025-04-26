#ifndef MATRIX_H
#define MATRIX_H
#include <vector>

namespace galanet {
    class Matrix {
        public:
            static constexpr double EPSILON = 1e-7;
            Matrix();
            Matrix(int num_rows, int num_cols);
            Matrix(int num_rows, int num_cols, double val);
            double &operator()(int i, int j);
            const double &operator()(int i, int j) const;
            Matrix(const Matrix &other);              // Copy constructor
            Matrix &operator=(const Matrix &other);   // Copy assignment operator
            Matrix(Matrix &&other) noexcept;          // Move constructor
            Matrix &operator=(Matrix &&other) noexcept; // Move assignment operator
            ~Matrix() = default;

            Matrix operator+(double scalar) const;
            friend Matrix operator+(double scalar, const Matrix &m); //overwrite inverse operation 
            Matrix operator-(double scalar) const;
            friend Matrix operator-(double scalar, const Matrix &m); //overwrite inverse operation 
            Matrix operator*(double scalar) const;
            friend Matrix operator*(double scalar, const Matrix &m); //overwrite inverse operation 
            Matrix operator/(double scalar) const;
            friend Matrix operator/(double scalar, const Matrix &m); //overwrite inverse operation 
            Matrix pow(int scalar) const;
            Matrix operator-() const;

            Matrix &operator+=(double scalar);
            Matrix &operator-=(double scalar);
            Matrix &operator*=(double scalar);
            Matrix &operator/=(double scalar);

            Matrix operator+(const Matrix &m) const;
            Matrix operator-(const Matrix &m) const;
            Matrix operator*(const Matrix &m) const;
            Matrix operator/(const Matrix &m) const;

            Matrix subset_rows(int start, int end) const;

            Matrix transpose() const;
            Matrix abs() const;
            Matrix sign() const;
            Matrix log() const;
            void fill(double value);
            double sum();
            int getCols() const;
            int getRows() const;
            std::vector<double> flatten() const;
            void print() const;
        private:
            std::vector<double> values;
            int num_rows;
            int num_cols;
    };
}
#endif