import os
import sys
import pyperclip
import subprocess

# Constantes globales
SEPARADOR = "=" * 50
ESPACIO_SIN_SEPARACION = "\u00A0"  # Carácter de espacio sin separación

def get_ignore_files(extra_files=None):
    """
    Retorna un conjunto con los nombres de archivos a ignorar.
    
    :param extra_files: Iterable con nombres adicionales a ignorar.
    :return: Conjunto con los nombres de archivos a ignorar.
    """
    base_files = {os.path.basename(__file__), "signals_2.py"}
    if extra_files:
        base_files.update(extra_files)
    return base_files

def get_ignore_dirs(extra_dirs=None):
    """
    Retorna un conjunto con los nombres de directorios a ignorar.
    
    :param extra_dirs: Iterable con nombres adicionales a ignorar.
    :return: Conjunto con los nombres de directorios a ignorar.
    """
    base_dirs = {"__pycache__"}
    if extra_dirs:
        base_dirs.update(extra_dirs)
    return base_dirs

def mostrar_estructura(ruta, ignore_files, ignore_dirs, prefijo=""):
    """
    Genera una cadena que representa la estructura del directorio, similar al comando 'tree'.
    Se ignoran los archivos y directorios indicados.
    
    :param ruta: Ruta del directorio a explorar.
    :param ignore_files: Conjunto de nombres de archivos a ignorar.
    :param ignore_dirs: Conjunto de nombres de directorios a ignorar.
    :param prefijo: Prefijo de indentación para la representación.
    :return: Cadena que representa la estructura del directorio.
    """
    salida = ""
    try:
        # Filtramos elementos ignorados, diferenciando archivos y directorios
        elementos = []
        for e in os.listdir(ruta):
            full_path = os.path.join(ruta, e)
            if os.path.isdir(full_path):
                if e in ignore_dirs:
                    continue
            elif os.path.isfile(full_path):
                if e in ignore_files:
                    continue
            elementos.append(e)
        elementos.sort()
        for i, elemento in enumerate(elementos):
            ruta_elemento = os.path.join(ruta, elemento)
            es_ultimo = (i == len(elementos) - 1)
            rama = "└── " if es_ultimo else "├── "
            salida += f"{prefijo}{rama}{elemento}\n"
            # Si es directorio y no se ignora, se recorre recursivamente
            if os.path.isdir(ruta_elemento):
                if elemento in ignore_dirs:
                    continue
                nuevo_prefijo = (prefijo + (ESPACIO_SIN_SEPARACION * 4)
                                  if es_ultimo else prefijo + "│" + ESPACIO_SIN_SEPARACION * 3)
                salida += mostrar_estructura(ruta_elemento, ignore_files, ignore_dirs, nuevo_prefijo)
    except FileNotFoundError:
        salida += f"Error: Ruta no encontrada: {ruta}\n"
    except PermissionError:
        salida += f"Error: Permiso denegado para acceder a: {ruta}\n"
    except Exception as e:
        salida += f"Error al listar el directorio {ruta}: {e}\n"
    return salida

def procesar_archivos(ruta_base, ignore_files, ignore_dirs, extension=".py"):
    """
    Procesa archivos con la extensión especificada dentro de la ruta base,
    ignorando archivos y directorios indicados, y retorna su contenido.
    
    :param ruta_base: Ruta base para la búsqueda.
    :param ignore_files: Conjunto de nombres de archivos a ignorar.
    :param ignore_dirs: Conjunto de nombres de directorios a ignorar.
    :param extension: Extensión de archivo a procesar (por defecto ".py").
    :return: Cadena con la información de los archivos procesados.
    """
    contenido_total = ""
    for ruta_actual, dirs, archivos in os.walk(ruta_base):
        # Excluir directorios que se deban ignorar
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for archivo in archivos:
            if archivo.endswith(extension) and archivo not in ignore_files:
                ruta_completa = os.path.join(ruta_actual, archivo)
                try:
                    with open(ruta_completa, "r", encoding="utf-8") as f:
                        contenido = f.read()
                    contenido_total += (
                        f"Archivo: {archivo}\n"
                        f"Ruta Completa: {ruta_completa}\n"
                        f"Contenido:\n{contenido}\n{SEPARADOR}\n"
                    )
                except FileNotFoundError:
                    print(f"Error: Archivo no encontrado: {ruta_completa}", file=sys.stderr)
                except Exception as e:
                    print(f"Error al leer el archivo {ruta_completa}: {e}", file=sys.stderr)
    return contenido_total

def copy_to_clipboard(texto):
    """
    Intenta copiar el texto dado al portapapeles utilizando pyperclip;
    si falla, se intenta con el comando 'pbcopy' (macOS).
    
    :param texto: Texto a copiar al portapapeles.
    :return: True si la copia fue exitosa, False en caso contrario.
    """
    try:
        pyperclip.copy(texto)
        return True
    except pyperclip.PyperclipException:
        try:
            subprocess.run("pbcopy", universal_newlines=True, input=texto, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

def main():
    # Configuración de elementos a ignorar
    ignore_files = get_ignore_files()          # Archivos a ignorar (por defecto: el script actual y "signals_2.py")
    ignore_dirs = get_ignore_dirs()            # Directorios a ignorar (por defecto: "__pycache__")
    
    # Se determina la ruta actual del script
    ruta_script = os.path.abspath(__file__)
    ruta_actual = os.path.dirname(ruta_script)
    print(f"Carpeta: {ruta_actual}")

    # Se genera la estructura de directorios y se procesan los archivos con extensión ".py"
    estructura = mostrar_estructura(ruta_actual, ignore_files, ignore_dirs)
    contenido_archivos = procesar_archivos(ruta_actual, ignore_files, ignore_dirs, extension=".py")
    contenido_completo = f"{estructura}{SEPARADOR}\n{contenido_archivos}"

    # Se intenta copiar la información al portapapeles
    if copy_to_clipboard(contenido_completo):
        print("La estructura del directorio y el contenido de los archivos han sido copiados al portapapeles.")
    else:
        print("Error: No se pudo copiar al portapapeles.")

    # Se muestra la estructura en consola
    print(SEPARADOR)
    print(estructura, end="")
    print(SEPARADOR)

if __name__ == "__main__":
    main()
