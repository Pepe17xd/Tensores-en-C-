#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <cmath>
#include <ctime>
using namespace std;

class TensorTransform;

class Tensor {
private:
    size_t shape[3];
    double *data;
    size_t total_size;
    int dimensiones;
    bool estado_view;
    friend class ReLU;
    friend class Sigmoid;

public:
    Tensor(const vector<size_t>& shape, const vector<double>& values):data(nullptr) {
        if (shape.size() > 3) {
            throw invalid_argument("Maximo solo puede ser de 3 dimensiones");
        }

        dimensiones = shape.size();
        total_size = 1;
        for (int i = 0; i < 3; i++)
            this->shape[i] = 1;

        for (size_t i = 0; i < shape.size(); i++) {
            this->shape[i] = shape[i];
            total_size *= shape[i];
        }

        if (values.size() != total_size) {
            throw invalid_argument("La cantidad de valores no coinciden con el tamano del array");
        }

        data = new double[total_size];
        estado_view = true;

        for (size_t i = 0; i < total_size; i++) {
            data[i] = values[i];
        }
    }

    static Tensor zeros(const vector<size_t>& shape) {
        size_t total = 1;
        for (size_t i = 0; i < shape.size(); i++) {
            total *= shape[i];
        }

        vector<double> values(total, 0.0);
        return Tensor(shape, values);
    }

    static Tensor ones(const vector<size_t>& shape) {
        size_t total = 1;
        for (size_t i = 0; i < shape.size(); i++) {
            total *= shape[i];
        }

        vector<double> values(total, 1.0);
        return Tensor(shape, values);
    }

    static Tensor random(const vector<size_t>& shape, double min, double max) {
        size_t total = 1;
        for (size_t i = 0; i < shape.size(); i++) {
            total *= shape[i];
        }

        vector<double> values(total);

        for (size_t i = 0; i < total; i++) {
            double r = (double)rand() / RAND_MAX;
            values[i] = min + r * (max - min);
        }

        return Tensor(shape, values);
    }

    static Tensor arange(int min, int max) {
        if (max <= min) {
            throw invalid_argument("max debe ser mayor que min");
        }

        size_t total = max - min;

        vector<size_t> shape = { total };
        vector<double> values(total);

        for (int i = 0; i < (int)total; i++) {
            values[i] = min + i;
        }

        return Tensor(shape, values);
    }

    // Prototipos requeridos en la clase Tensor
    //Constructor de copia
    Tensor(const Tensor& otra) {
        total_size = otra.total_size;
        dimensiones=otra.dimensiones;
        for (int i = 0; i < 3; i++) {
            shape[i] = otra.shape[i];
        }
        data = new double[total_size];
        estado_view=true;
        for (size_t i = 0; i < total_size; i++) {
            data[i] = otra.data[i];
        }
    }

    //Asignador de copia
    Tensor& operator=(const Tensor& otra) {
        if (this != &otra) {
            if (estado_view) {
                delete[] data;
            }
            total_size = otra.total_size;
            dimensiones=otra.dimensiones;

            for (int i = 0; i < 3; i++) {
                shape[i] = otra.shape[i];
            }
            data = new double[total_size];
            estado_view=true;
            for (size_t i = 0; i < total_size; i++) {
                data[i] = otra.data[i];
            }
        }
        return *this;
    }


    //constructor de movimiento
    Tensor(Tensor&& otra_clase) noexcept{
        total_size = otra_clase.total_size;
        dimensiones = otra_clase.dimensiones;
        for (int i = 0; i < 3; i++) {
            shape[i] = otra_clase.shape[i];
        }
        data = otra_clase.data;
        estado_view=otra_clase.estado_view;
        otra_clase.data = nullptr;
        otra_clase.total_size = 0;
        otra_clase.estado_view=false;
        for (int i = 0; i < 3; i++) {
            otra_clase.shape[i]=1;
        }
    }

    //Asinador de movimiento
    Tensor& operator=(Tensor&& otra_clase) noexcept {
        if (this != &otra_clase) {
            if (estado_view){
                delete[] data;
            }

            total_size = otra_clase.total_size;
            dimensiones = otra_clase.dimensiones;

            for (int i = 0; i < 3; i++) {
                shape[i] = otra_clase.shape[i];
            }

            data = otra_clase.data;
            estado_view=otra_clase.estado_view;

            otra_clase.data = nullptr;
            otra_clase.total_size = 0;
            otra_clase.estado_view=false;

            for (int i = 0; i < 3; i++) {
                otra_clase.shape[i]=1;
            }
        }
        return *this;
    }
    Tensor apply(const TensorTransform& transform) const;

    Tensor operator+(const Tensor& other) const {
        if (dimensiones != 2 || other.dimensiones != 2) {
            throw invalid_argument("Solo soporta tensores 2D");
        }

        size_t filas = shape[0];
        size_t cols  = shape[1];

        if (shape[0] == other.shape[0] && shape[1] == other.shape[1]) {

            vector<double> resultado(total_size);

            for (size_t i = 0; i < total_size; i++) {
                resultado[i] = data[i] + other.data[i];
            }

            return Tensor({filas, cols}, resultado);
        }

        if (other.shape[0] == 1 && other.shape[1] == cols) {

            vector<double> resultado(total_size);

            for (size_t i = 0; i < filas; i++) {
                for (size_t j = 0; j < cols; j++) {

                    size_t idx = i * cols + j;

                    resultado[idx] = data[idx] + other.data[j];
                }
            }

            return Tensor({filas, cols}, resultado);
        }

        throw invalid_argument("Dimensiones incompatibles");
    }

    Tensor operator-(const Tensor& other) const {
        for (int i = 0; i <3; i++) {
            if (shape[i] != other.shape[i]) {
                throw invalid_argument("Las dimensiones deben de ser igual, de lo contrario no se puede ejecutar la resta.");
            }
        }

        vector<double> suma_valores(total_size);

        for (size_t i = 0; i < total_size; i++) {
            suma_valores[i] = data[i]-other.data[i];
        }
        vector<size_t> resultado(shape,shape+3);
        return Tensor(resultado, suma_valores);
    }

    Tensor operator*(const Tensor& other) const {
        for (int i = 0; i <3; i++) {
            if (shape[i] != other.shape[i]) {
                throw invalid_argument("Para poder ejecutar la operacion los tensores deben de ser de la misma dimension");
            }
        }

        vector<double> suma_valores(total_size);

        for (size_t i = 0; i < total_size; i++) {
            suma_valores[i] = data[i]*other.data[i];
        }
        vector<size_t> resultado(shape,shape+3);
        return Tensor(resultado, suma_valores);
    }

    Tensor operator*( double escalar) const {
        vector<double> valores(total_size);
        for (size_t i = 0; i < total_size; i++) {
            valores[i] = data[i]*escalar;
        }
        vector<size_t> resultado(shape,shape+3);
        return Tensor(resultado, valores);
    }
    //metodo de impresion:
    void print() const {
        for (size_t i = 0; i < total_size; i++) {
            cout << data[i] << " ";
        }
        cout << endl;
    }

    //Metodo view
    Tensor(const vector<size_t>& new_shape, double* shared_data, size_t total) {
        total_size = total;
        dimensiones=new_shape.size();
        estado_view=false;
        for (int i = 0; i < 3; i++) {
            shape[i] = 1;
        }

        for (size_t i = 0; i < new_shape.size(); i++) {
            shape[i] = new_shape[i];
        }
        data = shared_data;
    }

    Tensor view(const vector<size_t>& new_shape) const {
        if ( new_shape.size()<1 || new_shape.size()>3 ) {
            throw invalid_argument("Hay un error");
        }

        size_t nuevo_total=1;
        for (size_t i = 0; i < new_shape.size(); i++) {
            nuevo_total *= new_shape[i];
        }

        if (nuevo_total != total_size) {
            throw invalid_argument("Hay un error en las dimensiones ");
        }

        return Tensor(new_shape, this->data, this->total_size);

    }

    // 7.2 unsqueeze sirve para agregar una dimension mas

    Tensor unsqueeze(int posicion) const {
        if (posicion<0 || posicion>dimensiones) {
            throw invalid_argument("Las posiciones no pueden ser negativas o mayores que 3");
        }
        //Siempre y cuando la poscion es la tercera, ya no se puede pq pasaria a la cuarta dimension

        if (dimensiones+1>3) {
            throw invalid_argument("No se puede agregar uno mas, ya que sobrepasa las dimensiones conocidas");
        }

        vector<size_t> mas_posicion;
        for (int i = 0; i < dimensiones+1; i++) {
            if (i==posicion) {
                mas_posicion.push_back(1);
            }
            else {
                mas_posicion.push_back(shape[i-(i>posicion)]);
            }
        }
        return Tensor(mas_posicion, this->data, total_size);
    }
    void print_shape() const {
        for (int i = 0; i < 3; i++) {
            cout << shape[i] << " ";
        }
        cout << endl;
    }

    // 8 concatenacion

    static Tensor concat(const Tensor& t1, const Tensor& t2, int dimension) {
        if (dimension<0 || dimension>t1.dimensiones) {
            throw invalid_argument("Dimensiones diferentes");
        }


        for (int i = 0; i < t1.dimensiones; i++) {
            if (i!=dimension) {
                if (t1.shape[i] != t2.shape[i]) {
                    throw invalid_argument("No coinciden");
                }
            }
        }

        vector<size_t> shape_concatenando_ambos;
        for (int i = 0; i < t1.dimensiones; i++) {
            if (i==dimension) {
                shape_concatenando_ambos.push_back(t1.shape[i]+t2.shape[i]);
            }
            else {
                shape_concatenando_ambos.push_back(t1.shape[i]);
            }
        }

        size_t nuevo_tamanio= t1.total_size+t2.total_size;
        double *nueva_data= new double[nuevo_tamanio];

        size_t copias=1;
        size_t bloques=1;
        for (int i = dimension+1; i < t1.dimensiones; i++) {
            copias *= t1.shape[i];
        }
        for (int i=0;i<dimension;i++) {
            bloques *= t1.shape[i];
        }

        size_t lugar=0;
        for (size_t b=0;b<bloques;b++) {
            size_t para1=b*t1.shape[dimension]*copias;
            size_t para2=b*t2.shape[dimension]*copias;

            for (size_t i=0;i<t1.shape[dimension]*copias;i++) {
                nueva_data[lugar++] = t1.data[para1+i];
            }
            for (size_t i=0;i<t2.shape[dimension]*copias;i++) {
                nueva_data[lugar++] = t2.data[para2+i];
            }
        }
        vector<double> valores(nuevo_tamanio);

        for (size_t i = 0; i < nuevo_tamanio; i++) {
            valores[i] = nueva_data[i];
        }

        delete[] nueva_data;

        return Tensor(shape_concatenando_ambos, valores);
    }

    friend Tensor dot(const Tensor& t1, const Tensor& t2) {
        if (t1.total_size!=t2.total_size) {
            throw invalid_argument("Los shape no son iguales");
        }

        double producto_punto=0;
        for (size_t i = 0; i < t1.total_size; i++) {
            producto_punto+=t1.data[i]*t2.data[i];
        }
        vector<double> valores={producto_punto};
        return Tensor ({1},valores);
    }

    friend Tensor matmul(const Tensor& t1, const Tensor& t2) {
        if (t1.dimensiones!=2 || t2.dimensiones!=2) {
            throw invalid_argument("Solo se pueden operar en matrices de 2 dimensiones");
        }

        if (t1.shape[1]!=t2.shape[0]) {
            throw invalid_argument("No se puede operar las matrices");
        }
        size_t m=t1.shape[0];
        size_t n=t1.shape[1];
        size_t p=t2.shape[1];

        vector<size_t> shape_resultado={m,p};
        vector<double> resultado(m*p,0.0);

        for (size_t i=0;i<m;i++) {
            for (size_t j=0;j<p;j++) {
                double sum=0;
                for (size_t k=0;k<n;k++) {
                    sum+=t1.data[i*n+k]*t2.data[k*p+j];
                }
                resultado[i*p+j]=sum;
            }
        }
        return Tensor (shape_resultado,resultado);
    }

    ~Tensor() {
        if (estado_view) {
            delete[] data;
        }
    }

};

class TensorTransform{
    public:
    virtual Tensor apply (const Tensor &t) const=0;
    virtual ~TensorTransform() =default;
};

//La clase TRasformacion del tensor y el metodo apply que se implmento fuera.
Tensor Tensor::apply(const TensorTransform &transform) const {
    return transform.apply(*this);
}

class ReLU : public TensorTransform {
public:
    Tensor apply (const Tensor &t) const override {
        Tensor resultado=t;
        for (size_t i = 0; i < resultado.total_size; i++) {
            resultado.data[i]=max(0.0, resultado.data[i]);
        }
        return resultado;
    }
};

class Sigmoid : public TensorTransform {
public:
    Tensor apply (const Tensor &t) const override {
        Tensor resultado=t;
        for (size_t i=0;i<resultado.total_size;i++) {
            resultado.data[i]=1.0/(1+exp(-resultado.data[i]));
        }
        return resultado;
    }
};


//El menu
int main() {
    srand(time(0));
    Tensor tensor_1 = Tensor::random({1000, 20, 20}, 0.0, 1.0);
    Tensor tensor1_view = tensor_1.view({1000, 400});
    Tensor tensor_2 = Tensor::random({400, 100}, -0.1, 0.1);
    Tensor aplicando_matmul = matmul(tensor1_view, tensor_2);
    Tensor tensor_3 = Tensor::random({1, 100}, -0.1, 0.1);
    Tensor aplicando_suma= aplicando_matmul + tensor_3;
    ReLU relu;
    Tensor aplicando_relu = aplicando_suma.apply(relu);
    Tensor tensor_4 = Tensor::random({100, 10}, -0.1, 0.1);
    Tensor otro_matmul = matmul(aplicando_relu, tensor_4);
    Tensor tensor_5 = Tensor::random({1, 10}, -0.1, 0.1);
    Tensor resultado_suma = otro_matmul + tensor_5;

    Sigmoid sigmoid;
    Tensor Red_neuronal = resultado_suma.apply(sigmoid);

    Red_neuronal.print();
    cout<<endl;
    return 0;
}