#include <iostream>
#include <cmath>
#include <algorithm>
#include "matrix.h"

namespace galanet{

        Matrix::Matrix() : num_rows(0), num_cols(0), values(){};
        Matrix::Matrix(int num_rows, int num_cols) : num_rows(num_rows), num_cols(num_cols), values(num_rows * num_cols, 0) {}
        Matrix::Matrix(int num_rows, int num_cols, double val) : num_rows(num_rows), num_cols(num_cols), values(num_rows * num_cols, val) {}
        //element-wise access
        double &Matrix::operator()(int i, int j) {
            if (i >= num_rows || j >= num_cols) throw std::invalid_argument("Index out of bounds");
            return values[i * num_cols + j];
        }
        const double &Matrix::operator()(int i, int j) const {
            if (i >= num_rows || j >= num_cols) throw std::invalid_argument("Index out of bounds");
            return values[i * num_cols + j];
        }


        //copy constructor
        Matrix::Matrix(const Matrix &other)
            : num_rows(other.num_rows), num_cols(other.num_cols), values(other.values) {}

        //copy assignment operator
        Matrix &Matrix::operator=(const Matrix &other) {
            if (this == &other) return *this; 
            num_rows = other.num_rows;
            num_cols = other.num_cols;
            values = other.values;
            return *this;
        }

        //move constructor
        Matrix::Matrix(Matrix &&other) noexcept
            : num_rows(other.num_rows), num_cols(other.num_cols), values(std::move(other.values)) {
            other.num_rows = 0;
            other.num_cols = 0;
        }

        //move assignment operator
        Matrix &Matrix::operator=(Matrix &&other) noexcept {
            if (this == &other) return *this; 
            num_rows = other.num_rows;
            num_cols = other.num_cols;
            values = std::move(other.values);
            other.num_rows = 0;
            other.num_cols = 0;
            return *this;
        }


        //matrix;scalar operations
        //addition
        Matrix Matrix::operator+(double scalar) const {
           Matrix result(num_rows, num_cols);
            std::transform(values.begin(), values.end(), result.values.begin(),
                   [scalar](double val) { return val + scalar; });
            return result;
        }

        Matrix operator+(double scalar, const Matrix &m) { 
            return m + scalar;
        }
        //subtraction
        Matrix Matrix::operator-(double scalar) const {
           Matrix result(num_rows, num_cols);
            std::transform(values.begin(), values.end(), result.values.begin(),
                   [scalar](double val) { return val - scalar; });
            return result;
        }

        Matrix operator-(double scalar, const Matrix &m) { 
            return m - scalar;
        }
        //multiplication
        Matrix Matrix::operator*(double scalar) const {
            Matrix result(num_rows, num_cols);
            std::transform(values.begin(), values.end(), result.values.begin(),
                   [scalar](double val) { return val * scalar; });
            return result;
        }

        Matrix operator*(double scalar, const Matrix &m) { 
            return m * scalar;
        }
        //divison
        Matrix Matrix::operator/(double scalar) const {
            Matrix result(num_rows, num_cols);
            std::transform(values.begin(), values.end(), result.values.begin(),
                   [scalar](double val) { return val / scalar; });
            return result;
        }

        Matrix operator/(double scalar, const Matrix &m) { 
            return m / scalar;
        }

        //pow
        Matrix Matrix::pow(int scalar) const{
           Matrix result(num_rows, num_cols);
            std::transform(values.begin(), values.end(), result.values.begin(),
                   [scalar](double val) { return std::pow(val, scalar); });
            return result;
        }
        Matrix Matrix::operator-() const{
            Matrix result(num_rows, num_cols);
            std::transform(values.begin(), values.end(), result.values.begin(),
                   [](double val) { return -val; });
            return result;
        }

        //self operations
        Matrix &Matrix::operator+=(double scalar) { // Scalar addition assignment
            std::transform(values.begin(), values.end(), values.begin(),
                   [scalar](double val) { return val + scalar; });
            return *this;
        }
        Matrix &Matrix::operator-=(double scalar) { // Scalar subtraction assignment
            std::transform(values.begin(), values.end(), values.begin(),
                   [scalar](double val) { return val - scalar; });
            return *this;
        }
        Matrix &Matrix::operator*=(double scalar) { // Scalar multiplication assignment
            std::transform(values.begin(), values.end(), values.begin(),
                   [scalar](double val) { return val * scalar; });
            return *this;
        }
        Matrix &Matrix::operator/=(double scalar) { // Scalar division assignment
            std::transform(values.begin(), values.end(), values.begin(),
                   [scalar](double val) { return val / scalar; });
            return *this;
        }
        //matrix;matrix operations
        //addition
        Matrix Matrix::operator+(const Matrix &m) const{
            if(num_cols!=m.num_cols || num_rows!=m.num_rows) throw std::invalid_argument("Shape not compatible for addition");
            Matrix res(num_rows,num_cols);
            #pragma omp parallel for
            for (size_t i = 0; i < num_rows * num_cols; i++) 
                res.values[i] = values[i] + m.values[i];
            return res;
           
        }
        //subtraction
        Matrix Matrix::operator-(const Matrix &m) const{
            if(num_cols!=m.num_cols || num_rows!=m.num_rows) throw std::invalid_argument("Shape not compatible for subtraction");
            Matrix res(num_rows,num_cols);
            #pragma omp parallel for
            for (size_t i = 0; i < num_rows * num_cols; i++) 
                res.values[i] = values[i] - m.values[i];
            return res;
        }
        //multiplication(dot product)
        Matrix Matrix::operator*(const Matrix &m) const{
            if(num_cols!=m.num_rows)throw std::invalid_argument("Shape not compatible for matrix multiplication");
            Matrix res(num_rows, m.num_cols,0);
            for (size_t i = 0; i < num_rows; i++) {
                for (size_t j = 0; j < m.num_cols; j++) 
                    for (size_t k = 0; k < num_cols; k++) {
                        res(i, j) += values[i * num_cols + k] * m(k, j);
                    }
            }

            return res;
        }
        //subtraction
        Matrix Matrix::operator/(const Matrix &m) const{
            if(num_cols!=m.num_cols || num_rows!=m.num_rows) throw std::invalid_argument("Shape not compatible for divison");
            Matrix res(num_rows,num_cols);
            #pragma omp parallel for
            for (size_t i = 0; i < num_rows * num_cols; i++) 
                res.values[i] = values[i] / m.values[i];
            return res;
        }


        Matrix Matrix::subset_rows(int start, int end) const{
            if(start<0 || end>num_rows) throw std::invalid_argument("Index out of bounds");
            Matrix res(end-start,num_cols);
            for(size_t i=start;i<end;i++)
                for(size_t j=0;j<num_cols;j++)
                    res(i-start,j)=values[i*num_cols+j];
            return res;
        }

        //unary operations
        //transpose
        Matrix Matrix::transpose() const{
            Matrix res(num_cols,num_rows);
            for(size_t i=0;i<num_rows;i++)
                for(size_t j=0;j<num_cols;j++)
                    res(j,i)=this->operator()(i,j);
            return res;
        }
        //abs
        Matrix Matrix::abs() const{
            Matrix res(num_rows,num_cols);
            #pragma omp parallel for
            for (size_t i = 0; i < num_rows * num_cols; i++) 
                res.values[i] = std::abs(values[i]);
            return res;
        }
        //sign 
        Matrix Matrix::sign() const {
            Matrix res(num_rows, num_cols);
            #pragma omp parallel for
            for (size_t i = 0; i < num_rows * num_cols; i++) {
                res.values[i] = std::copysign(1.0, values[i]);
            }
            return res;
        }
        //log
        Matrix Matrix::log() const {
            Matrix res(num_rows, num_cols);
            #pragma omp parallel for
            for (size_t i = 0; i < num_rows * num_cols; i++) {
                if (values[i] < EPSILON) {
                    throw std::domain_error("Log undefined for values <= 0");
                }
                res.values[i] = std::log(values[i] + EPSILON);
            }
            return res;
    }
        //fill
        void Matrix::fill(double value) {
            for (size_t i = 0; i < num_rows * num_cols; i++) 
                values[i] = value;
            
        }
        //sum
        double Matrix::sum(){
            double s=0;
            #pragma omp parallel for reduction(+:s)
            for (size_t i = 0; i < num_rows * num_cols; i++) 
                s+=values[i];
            return s;
        }
    

        //getters
        int Matrix::getRows() const {
            return num_rows;
        }

        int Matrix::getCols() const {
            return num_cols;
        }
        std::vector<double> Matrix::flatten() const {
            return values;
        }
        //print
        void Matrix::print() const{
            for(size_t i=0;i<num_rows;i++){
                for(size_t j=0;j<num_cols;j++)
                    std::cout<<values[i*num_cols+j]<<" ";
                std::cout<<"\n";
            }
        }

}