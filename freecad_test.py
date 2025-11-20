import os
import sys

def init_freecad():
    """
    Auto-detects FreeCAD folder placed next to this script.
    Expected structure:
        <project_root>/
            FreeCAD/
                bin/
                lib/
                Mod/
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    freecad_base = os.path.join(script_dir, "FreeCAD")

    bin_path = os.path.join(freecad_base, "bin")
    lib_path = os.path.join(freecad_base, "lib")
    mod_path = os.path.join(freecad_base, "Mod")

    print("Looking for FreeCAD in:", freecad_base)
    print("BIN exists:", os.path.isdir(bin_path))
    print("LIB exists:", os.path.isdir(lib_path))
    print("MOD exists:", os.path.isdir(mod_path))

    # Ensure folder exists
    if not os.path.isdir(bin_path):
        raise RuntimeError(f"ERROR: FreeCAD/bin not found at: {bin_path}")
    if not os.path.isdir(lib_path):
        raise RuntimeError(f"ERROR: FreeCAD/lib not found at: {lib_path}")
    if not os.path.isdir(mod_path):
        raise RuntimeError(f"ERROR: FreeCAD/Mod not found at: {mod_path}")

    # Add bin so Python can import FreeCAD.pyd
    sys.path.append(bin_path)

    # Add bin+lib to PATH so DLLs load
    os.environ["PATH"] = bin_path + ";" + lib_path + ";" + os.environ.get("PATH", "")

    # Add all Mod subpackages to sys.path
    for folder in os.listdir(mod_path):
        full = os.path.join(mod_path, folder)
        if os.path.isdir(full):
            sys.path.append(full)

    # Try imports
    try:
        import FreeCAD
        import Part
    except Exception as e:
        raise RuntimeError(f"Failed to import FreeCAD modules: {e}")

    print("FreeCAD core modules imported OK.")
    return True


# -----------------------------------------------
# RUN TEST
# -----------------------------------------------

if __name__ == "__main__":
    try:
        init_freecad()
    except Exception as e:
        print("\nINIT FAILED:", e)
        print("\nMake sure your folder structure is:")
        print("  <project>/FreeCAD/bin")
        print("  <project>/FreeCAD/lib")
        print("  <project>/FreeCAD/Mod")
        raise SystemExit(1)

    import Part

    test_file = "test.step"
    if not os.path.exists(test_file):
        print("\nPlace a file named 'test.step' next to this script to test geometry loading.\n")
        raise SystemExit(1)

    try:
        shape = Part.Shape()
        shape.read(test_file)
    except Exception as e:
        print("ERROR reading STEP file:", e)
        raise

    print("\nSTEP LOADED SUCCESSFULLY!")
    print("Volume:", shape.Volume)
    print("Area:", shape.Area)
    print("BBox:", shape.BoundBox)
    print("Faces:", len(shape.Faces))
    print("Edges:", len(shape.Edges))
    print("\nAll good.")
