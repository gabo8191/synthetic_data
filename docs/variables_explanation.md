# Explicación de Variables del Dataset Titanic

Este documento explica el significado y características de cada variable en el dataset Titanic utilizado para la generación de datos sintéticos.

## Variables del Dataset

### 1. **PassengerId** (ID del Pasajero)

- **Tipo**: Numérico entero
- **Descripción**: Identificador único para cada pasajero
- **Rango**: 1 a 891
- **Uso en el modelo**: Se excluye del entrenamiento (variable identificadora)

### 2. **Survived** (Sobrevivió)

- **Tipo**: Binario (0/1)
- **Descripción**: Variable objetivo que indica si el pasajero sobrevivió al hundimiento
- **Valores**:
  - `0`: No sobrevivió
  - `1`: Sobrevivió
- **Distribución original**: 61.6% no sobrevivieron, 38.4% sobrevivieron
- **Uso en el modelo**: Variable objetivo para entrenamiento estratificado

### 3. **Pclass** (Clase del Pasajero)

- **Tipo**: Numérico entero
- **Descripción**: Clase socioeconómica del pasajero
- **Valores**:
  - `1`: Primera clase (alta)
  - `2`: Segunda clase (media)
  - `3`: Tercera clase (baja)
- **Distribución**: 24.2% primera, 20.7% segunda, 55.1% tercera
- **Uso en el modelo**: Variable numérica con alta precisión (4.2% diferencia)

### 4. **Name** (Nombre)

- **Tipo**: Texto
- **Descripción**: Nombre completo del pasajero
- **Ejemplo**: "Braund, Mr. Owen Harris"
- **Uso en el modelo**: Se excluye del entrenamiento (variable identificadora)

### 5. **Sex** (Género)

- **Tipo**: Categórica
- **Descripción**: Género del pasajero
- **Valores**:
  - `male`: Masculino
  - `female`: Femenino
- **Distribución**: 64.8% masculino, 35.2% femenino
- **Uso en el modelo**: Variable categórica con buena discriminación

### 6. **Age** (Edad)

- **Tipo**: Numérico decimal
- **Descripción**: Edad del pasajero en años
- **Rango**: 0.42 a 80 años
- **Media**: 29.36 años
- **Uso en el modelo**: Variable numérica con precisión moderada (22.9% diferencia)

### 7. **SibSp** (Hermanos/Cónyuges)

- **Tipo**: Numérico entero
- **Descripción**: Número de hermanos o cónyuges a bordo
- **Rango**: 0 a 8
- **Media**: 0.52
- **Uso en el modelo**: Variable de conteo discreto con baja precisión (198.8% diferencia)

### 8. **Parch** (Padres/Hijos)

- **Tipo**: Numérico entero
- **Descripción**: Número de padres o hijos a bordo
- **Rango**: 0 a 6
- **Media**: 0.38
- **Uso en el modelo**: Variable de conteo discreto con baja precisión (56.4% diferencia)

### 9. **Ticket** (Número de Boleto)

- **Tipo**: Texto
- **Descripción**: Número de boleto del pasajero
- **Ejemplo**: "A/5 21171", "PC 17599"
- **Uso en el modelo**: Se excluye del entrenamiento (variable identificadora)

### 10. **Fare** (Tarifa)

- **Tipo**: Numérico decimal
- **Descripción**: Precio pagado por el boleto
- **Rango**: 0 a 512.33
- **Media**: 32.20
- **Uso en el modelo**: Variable numérica con baja precisión (52.0% diferencia)

### 11. **Cabin** (Cabina)

- **Tipo**: Texto
- **Descripción**: Número de cabina del pasajero
- **Ejemplo**: "C85", "B96 B98"
- **Uso en el modelo**: Se excluye del entrenamiento (variable identificadora)

### 12. **Embarked** (Puerto de Embarque)

- **Tipo**: Categórica
- **Descripción**: Puerto donde embarcó el pasajero
- **Valores**:
  - `C`: Cherbourg
  - `Q`: Queenstown
  - `S`: Southampton
- **Distribución**: 18.9% C, 8.6% Q, 72.5% S
- **Uso en el modelo**: Variable categórica con discriminación crítica (colapso hacia S)

## Variables Utilizadas en el Modelo

### Variables de Entrada (Features)

Después del preprocesamiento, las siguientes variables se utilizan para entrenar el GAN:

1. **pclass** - Clase del pasajero (escalada)
2. **age** - Edad (escalada)
3. **sibsp** - Hermanos/cónyuges (escalada)
4. **parch** - Padres/hijos (escalada)
5. **fare** - Tarifa (escalada)
6. **sex_female** - Género femenino (one-hot)
7. **sex_male** - Género masculino (one-hot)
8. **embarked_C** - Puerto Cherbourg (one-hot)
9. **embarked_Q** - Puerto Queenstown (one-hot)
10. **embarked_S** - Puerto Southampton (one-hot)

### Variable Objetivo

- **survived** - Usada para entrenamiento estratificado (un GAN por clase)

## Preprocesamiento Aplicado

### 1. **Exclusión de Variables Identificadoras**

- `passengerid`, `name`, `ticket`, `cabin` se excluyen del entrenamiento

### 2. **One-Hot Encoding**

- Variables categóricas (`sex`, `embarked`) se convierten a variables binarias

### 3. **Estandarización**

- Variables numéricas continuas se escalan usando `StandardScaler`

### 4. **Normalización para GAN**

- Todos los datos se normalizan al rango [-1, 1] para compatibilidad con `Tanh`

## Métricas de Calidad por Variable

| Variable | Tipo       | Precisión        | Discriminación | Problema Principal  |
| -------- | ---------- | ---------------- | -------------- | ------------------- |
| pclass   | Numérica   | Alta (4.2%)      | Excelente      | Ninguno             |
| age      | Numérica   | Moderada (22.9%) | Buena          | Sesgo hacia jóvenes |
| sibsp    | Numérica   | Crítica (198.8%) | Pobre          | Sobreestimación     |
| parch    | Numérica   | Baja (56.4%)     | Pobre          | Sobreestimación     |
| fare     | Numérica   | Baja (52.0%)     | Moderada       | Sobreestimación     |
| sex      | Categórica | Buena            | Buena          | Sesgo menor         |
| embarked | Categórica | Crítica          | Crítica        | Colapso hacia S     |

## Notas Importantes

1. **Variables de Conteo**: `sibsp` y `parch` son particularmente desafiantes para GANs simples debido a su naturaleza discreta
2. **Variables Desbalanceadas**: `embarked` muestra colapso hacia la categoría dominante (S)
3. **Variables Continuas**: `age` y `fare` muestran sesgos sistemáticos en la generación
4. **Variables Simples**: `pclass` y `sex` muestran excelente fidelidad en la generación
