"""
Napari plugin reader for bioio.
"""

import os
from typing import Any

import bioio


def napari_get_reader(path: str):
    """Return a reader function for napari.

    The reader will be default bioio if supported, otherwise using bioio-bioformats.

    Parameters
    ----------
    path : str
        Path to the file or directory.

    Returns
    -------
    callable or None
        A function that returns layer data, or None if the path is not supported.

    Examples
    --------
    >>> reader = napari_get_reader('image.tif')
    >>> if reader is not None:
    ...     data = reader('image.tif')
    ...     # Use data in napari
    """
    # Try default bioio reader
    try:
        if bioio.bio_image.BioImage.is_supported_image(path):
            return bioio_napari_reader
    except (AttributeError, ImportError):
        pass
    # Try bioio-bioformats as fallback
    try:
        import bioio_bioformats

        if bioio_bioformats.Reader.is_supported_image(path):
            return bioio_napari_reader
    except (AttributeError, ImportError):
        pass
    return None


def bioio_napari_reader(path: str) -> list[Any]:
    """
    Read image data from the given path using bioio and return napari layer data.

    Returns all scenes in the image as individual napari layers.
    Automatically tries default bioio first, then falls back to bioio-bioformats.
    Uses actual scene names from metadata when available, otherwise falls back to numbered scenes.

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    list[tuple[Any, dict, str]]
        List of tuples: (data, metadata, layer_type) for each scene.
        Each scene becomes a separate layer in napari.

        For multi-scene images, metadata contains:
        - 'bioio_metadata': Original bioio metadata
        - 'scene_info': Dict with scene_id, scene_index, scene_name, and total_scenes

    Examples
    --------
    Read a single-scene image:

    >>> layers = bioio_napari_reader('single_scene.tif')
    >>> len(layers)  # 1
    1
    >>> data, meta, layer_type = layers[0]
    >>> layer_type
    'image'

    Read a multi-scene image with named scenes:

    >>> layers = bioio_napari_reader('multi_scene.czi')
    >>> len(layers)  # Number of scenes in the file
    3
    >>> for i, (data, meta, layer_type) in enumerate(layers):
    ...     print(f"Layer: {meta['name']}")
    ...     scene_info = meta['metadata']['scene_info']
    ...     print(f"Scene name: {scene_info['scene_name']}")
    Layer: multi_scene.czi - Tumor Region
    Scene name: Tumor Region
    Layer: multi_scene.czi - Control Region
    Scene name: Control Region
    Layer: multi_scene.czi - Background
    Scene name: Background
    """
    def _extract_scene_name(img, scene_id: str, scene_idx: int) -> str:
        """
        Extract scene name from metadata, fallback to numbered name.
        
        Parameters
        ----------
        img : BioImage
            The bioio image object with scene set.
        scene_id : str
            The scene identifier.
        scene_idx : int
            The scene index.
            
        Returns
        -------
        str
            The scene name or fallback name.
        """
        try:
            # Try to get scene name from OME metadata
            if hasattr(img, 'ome_metadata') and img.ome_metadata:
                ome = img.ome_metadata
                if hasattr(ome, 'images') and ome.images:
                    if scene_idx < len(ome.images):
                        image_meta = ome.images[scene_idx]
                        # Try different possible name attributes
                        if hasattr(image_meta, 'name') and image_meta.name:
                            return image_meta.name
                        if hasattr(image_meta, 'id') and image_meta.id and image_meta.id != scene_id:
                            # Use ID if it's different from the generic scene_id
                            return image_meta.id
            
            # Try to extract from scene_id if it contains meaningful info
            if scene_id and '#' in scene_id:
                # Format like "filename.czi #Scene_Name" or "filename.czi #01"
                parts = scene_id.split('#', 1)
                if len(parts) > 1:
                    scene_part = parts[1].strip()
                    # If it's not just a number, use it as name
                    if not scene_part.isdigit() and scene_part:
                        return scene_part
            
            # Try to get from current metadata
            if hasattr(img, 'metadata') and img.metadata:
                meta = img.metadata
                # Look for common name fields in metadata
                for name_field in ['name', 'title', 'scene_name', 'image_name']:
                    if hasattr(meta, name_field):
                        name_value = getattr(meta, name_field)
                        if name_value and isinstance(name_value, str):
                            return name_value
        
        except Exception:
            # If anything fails, fall back to numbered name
            pass
        
        # Fallback to numbered scene
        return f"Scene {scene_idx}"
    
    base_name = os.path.basename(path)
    layers = []
    img = None

    # Try default bioio reader first
    try:
        img = bioio.bio_image.BioImage(path)
    except Exception:
        # Fall back to bioio-bioformats
        try:
            import bioio_bioformats
            img = bioio.BioImage(path, reader=bioio_bioformats.Reader)
        except Exception as e:
            raise RuntimeError(f"Failed to read image with both bioio and bioio-bioformats: {e}")

    # Get all available scenes
    available_scenes = img.scenes

    # If only one scene, use the original naming
    if len(available_scenes) == 1:
        data = img.data
        meta = {"name": base_name, "metadata": img.metadata}
        layers.append((data, meta, "image"))
    else:
        # Multiple scenes: create a layer for each scene
        for scene_idx, scene_id in enumerate(available_scenes):
            img.set_scene(scene_id)
            data = img.data
            
            # Extract meaningful scene name
            scene_name = _extract_scene_name(img, scene_id, scene_idx)
            layer_name = f"{base_name} - {scene_name}"
            
            meta = {
                "name": layer_name,
                "metadata": {
                    "bioio_metadata": img.metadata,
                    "scene_info": {
                        "scene_id": scene_id,
                        "scene_index": scene_idx,
                        "scene_name": scene_name,
                        "total_scenes": len(available_scenes),
                    }
                }
            }
            layers.append((data, meta, "image"))

    return layers
