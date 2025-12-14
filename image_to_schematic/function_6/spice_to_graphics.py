from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os
import re
import tempfile

from spicelib import SpiceEditor  # type: ignore


@dataclass(frozen=True)
class DeviceNode:
    ref: str
    kind: str
    nodes: List[str]


@dataclass(frozen=True)
class CircuitGraph:
    devices: List[DeviceNode]
    net_to_pins: Dict[str, List[Tuple[str, int]]]


def _normalize_net_name(net: str) -> str:
    s = net.strip()
    low = s.lower()
    if low in {"0", "gnd", "ground"}:
        return "GND"
    return s


def _net_id(net_name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", net_name)
    return f"net_{safe}"


def _select_circuit(editor: SpiceEditor, subckt_name: Optional[str]) -> "SpiceEditor":
    """
    Decide which circuit to inspect:
    - if subckt_name is given -> return that subcircuit
    - else if main netlist has components -> use main
    - else if single subckt exists -> use that
    - else -> raise
    """
    if subckt_name:
        circuit = editor.get_subcircuit_named(subckt_name)
        if circuit is None:
            raise ValueError(f"Subcircuit '{subckt_name}' not found in netlist.")
        return circuit  # SpiceCircuit

    # Try main netlist first
    root_refs = editor.get_components(prefixes="*")
    if root_refs:
        return editor  # use main netlist

    # No main components: fall back to subcircuits
    names = editor.get_subcircuit_names()
    if not names:
        raise ValueError("Netlist contains no components and no subcircuits to inspect.")

    if len(names) == 1:
        # Only one subcircuit -> unambiguous
        return editor.get_subcircuit_named(names[0])

    # Ambiguous: user must specify which subcircuit they want
    raise ValueError(
        "Netlist has multiple subcircuits and no main-level components. "
        f"Available subcircuits: {', '.join(names)}. "
        "Please specify subckt_name."
    )


def _build_circuit_graph_from_spicelib(
    spice_code: str=None,
    subckt_name: Optional[str] = None,
) -> CircuitGraph:
    tmp_path = "bq79616_subckt_002_s37.cir"
    try:
        # with tempfile.NamedTemporaryFile(suffix=".net", delete=False, mode="w", encoding="utf-8") as f:
        #     tmp_path = f.name
        #     f.write(spice_code)

        editor = SpiceEditor(tmp_path)

        circuit = _select_circuit(editor, subckt_name)

        # Now we are on the correct SpiceCircuit (editor or subckt)
        refs = circuit.get_components(prefixes="*")
        refs = sorted(refs, key=lambda r: (r[0].upper() if r else "", r))

        devices: List[DeviceNode] = []
        for ref in refs:
            nodes_raw = circuit.get_component_nodes(ref)
            nodes = [_normalize_net_name(n) for n in nodes_raw]
            kind = ref[0].upper() if ref else "?"
            devices.append(DeviceNode(ref=ref, kind=kind, nodes=nodes))

        net_to_pins: Dict[str, List[Tuple[str, int]]] = {}
        for dev in devices:
            for pin_idx, net_name in enumerate(dev.nodes):
                net_to_pins.setdefault(net_name, []).append((dev.ref, pin_idx))

        for net, pins in net_to_pins.items():
            pins.sort(key=lambda x: (x[0], x[1]))

        return CircuitGraph(devices=devices, net_to_pins=net_to_pins)

    except Exception as e:
        raise e
    # finally:
    #     if tmp_path is not None and os.path.exists(tmp_path):
    #         try:
    #             os.remove(tmp_path)
    #         except OSError:
    #             pass


def spice_to_mermaid(spice_code: str=None, subckt_name: Optional[str] = None) -> str:
    graph = _build_circuit_graph_from_spicelib(spice_code, subckt_name=subckt_name)

    lines: List[str] = []
    lines.append("graph LR")

    # Devices
    for dev in graph.devices:
        label = f"{dev.ref} ({dev.kind})"
        lines.append(f'    {dev.ref}["{label}"]')

    # Nets
    net_names = sorted(graph.net_to_pins.keys())
    for net in net_names:
        node_id = _net_id(net)
        label = net
        lines.append(f'    {node_id}["{label}"]')

    # Edges
    edge_lines: List[str] = []
    for net in net_names:
        net_node = _net_id(net)
        for (ref, pin_idx) in graph.net_to_pins[net]:
            edge_lines.append(f"    {net_node} --- {ref}")

    edge_lines.sort()
    lines.extend(edge_lines)

    return "\n".join(lines)


def spice_to_dot(
    spice_code: str=None,
    subckt_name: Optional[str] = None,
    rankdir: str = "LR",
) -> str:
    graph = _build_circuit_graph_from_spicelib(spice_code, subckt_name=subckt_name)

    lines: List[str] = []
    lines.append("graph G {")
    lines.append(f'  graph [rankdir="{rankdir}"];')
    lines.append('  node [fontname="Helvetica"];')

    for dev in graph.devices:
        label = f"{dev.ref}\\n({dev.kind})"
        lines.append(f'  "{dev.ref}" [shape=box,label="{label}"];')

    net_names = sorted(graph.net_to_pins.keys())
    for net in net_names:
        node_id = _net_id(net)
        label = net
        lines.append(f'  "{node_id}" [shape=circle,label="{label}"];')

    edge_lines: List[str] = []
    for net in net_names:
        net_node = _net_id(net)
        for (ref, pin_idx) in graph.net_to_pins[net]:
            edge_lines.append(f'  "{net_node}" -- "{ref}";')

    edge_lines.sort()
    lines.extend(edge_lines)

    lines.append("}")
    return "\n".join(lines)

def spice_to_svg(
    spice_code: str,
    subckt_name: Optional[str] = None,
    rankdir: str = "LR",
) -> str:
    from graphviz import Source  # raises if not installed

    dot = spice_to_dot(spice_code, subckt_name=subckt_name, rankdir=rankdir)
    src = Source(dot)
    svg_bytes = src.pipe(format="svg")
    return svg_bytes.decode("utf-8")


if __name__ == "__main__":
    spice = """
* Simple RC
V1 in 0 DC 5
R1 in out 10k
C1 out 0 1u
.end
"""

    # print("=== Mermaid ===")
    # print(spice_to_mermaid(spice))
    # print()

    # print("=== DOT ===")
    # print(spice_to_dot(spice))
    mermaid = spice_to_mermaid(subckt_name="BQ79616_stub")
    print(mermaid)

    dot = spice_to_dot(subckt_name="BQ79616_stub")
    print(dot)


    # If graphviz is installed:
    # svg = spice_to_svg(spice)
    # with open("rc_schematic.svg", "w", encoding="utf-8") as f:
    #     f.write(svg)
