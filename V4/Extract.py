import os
import pyperclip

separador = "=================================================="
espacio_sin_separacion = "\u00A0"  # Carácter de espacio sin separación


def mostrar_estructura(ruta, prefijo="", salida=""):
    """Muestra la estructura de directorios similar al comando tree y devuelve la cadena."""
    try:
        elementos = [e for e in os.listdir(ruta) if e != os.path.basename(__file__)]
        for i, elemento in enumerate(elementos):
            ruta_elemento = os.path.join(ruta, elemento)
            es_ultimo = i == len(elementos) - 1
            prefijo_actual = prefijo + ("└── " if es_ultimo else "├── ")
            salida += prefijo_actual + elemento + "\n"
            if os.path.isdir(ruta_elemento):
                prefijo_siguiente = prefijo + (
                    espacio_sin_separacion * 4
                    if es_ultimo
                    else "│" + espacio_sin_separacion * 3
                )  # Usa espacios sin separación
                salida = mostrar_estructura(ruta_elemento, prefijo_siguiente, salida)
        return salida
    except FileNotFoundError:
        return f"Error: Ruta no encontrada: {ruta}\n"
    except PermissionError:
        return f"Error: Permiso denegado para acceder a: {ruta}\n"
    except Exception as e:
        return f"Error al listar el directorio {ruta}: {e}\n"


def procesar_archivos(ruta_base):
    """Procesa archivos .css, .tsx y .js y devuelve el contenido como cadena."""
    contenido_total = ""
    try:
        for ruta_actual, _, archivos in os.walk(ruta_base):
            for archivo in archivos:
                if archivo.endswith((".py")) and archivo != os.path.basename(__file__):
                    ruta_completa = os.path.join(ruta_actual, archivo)
                    try:
                        with open(ruta_completa, "r", encoding="utf-8") as f:
                            contenido = f.read()
                            contenido_total += f"Archivo: {archivo}\nRuta Completa: {ruta_completa}\nContenido:\n{contenido}\n{separador}\n"
                    except FileNotFoundError:
                        print(f"Error: Archivo no encontrado: {ruta_completa}")
                    except Exception as e:
                        print(f"Error al leer el archivo {ruta_completa}: {e}")
    except Exception as e:
        print(f"Error al procesar la carpeta: {e}")
    return contenido_total


if __name__ == "__main__":
    ruta_script = os.path.abspath(__file__)
    ruta_actual = os.path.dirname(ruta_script)
    print(f"Carpeta: {ruta_actual}")
    salida_tree = mostrar_estructura(ruta_actual)
    contenido_archivos = procesar_archivos(ruta_actual)
    contenido_completo = salida_tree + separador + "\n" + contenido_archivos
    try:
        pyperclip.copy(contenido_completo)
        print(
            "La estructura del directorio y el contenido de los archivos han sido copiados al portapapeles."
        )
    except pyperclip.PyperclipException:
        try:
            import subprocess

            subprocess.run("pbcopy", universal_newlines=True, input=contenido_completo)
            print(
                "La estructura del directorio y el contenido de los archivos han sido copiados al portapapeles (usando pbcopy, macOS)."
            )
        except FileNotFoundError:
            print(
                "Error: No se pudo copiar al portapapeles. Asegúrate de tener pyperclip instalado (pip install pyperclip) o un sistema compatible con pbcopy (macOS)."
            )
    print(separador)
    print(salida_tree, end="")
    print(separador)
