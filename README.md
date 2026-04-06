Instrucciones de compilacion y ejecucion de la clase tensor

El código proporciona un sistema de tensores que tiene características como:
- Tensores de maximo 3 dimensiones
- Uso de memoria dinamica
- Constructor de copia
- Asignación por copia
- Constructor de movimiento
- Asignación por movimiento
- Uso de view para evitar copiar datos
- Transformaciones mediante polimorfismo

El tensor permite realizar las siguientes operaciones:
- Suma de tensores 
- Resta de tensores
- Multiplicación de elemento a elemento
- Multiplicación escalar.
  
Tambien se implementaron las siguientes funciones:

- `matmul`: multiplica matrices con tensores 2D
- `dot`: producto punto entre tensores
- `concat`: concatenación de tensores en una dimensión concreta
- `view`: interpretar dimensiones sin copiar datos
- `unsqueeze`: agrega una dimensión con tamaño 1.

El codigo cuenta con lo siguiente tensores pre definidos:

- `zeros(shape)`: tensor lleno de ceros  
- `ones(shape)`: tensor lleno de unos  
- `random(shape, min, max)`: valores aleatorios en un rango  
- `arange(min, max)`: secuencia de valores.  


Se implementa una clase base y las clases derivadas como: 
- ReLU
- Sigmoid
Para su aplicacion, se usa tensor.apply(transform)

Para el main se implementa un red neuronal simple que genera un tensor inicial de de dimensiones (1000 x 20 x 20), se trasnforma con view a (1000 x 400) y se le multiplica de forma matricial con un tensor de (1000 x 400) y se añade bias  (1 x 100) y se le aplica la funcion ReLU. Des pues de ello se aplica una multiplicacion matricial con tensor (100 x 10) y a eso se se añade bias de (1 x 10), y se le aplica la funcion sigmoid y finalmente se imprime el resultado. 
