#pragma once

#include <graphviz/gvc.h>
#include <graphviz/cgraph.h>
#include <string>
#include "autodiff.hpp"

namespace utils {

template <typename T>
static void
build_graph(
    autodiff::Node<T> const & node,
    autodiff::Tape<T> & tape,
    Agraph_t *g)
{
    if(node.idx == 0 /* || node.op == autodiff::Op::NOP */) {
        return;
    }

    std::string cur_id =
        std::to_string(reinterpret_cast<uintptr_t>(&node));
    Agnode_t *n_cur =
        agnode(g, const_cast<char*>(cur_id.c_str()), true);
    agsafeset(n_cur,
        const_cast<char*>("shape"),
        const_cast<char*>("Mrecord"),
        const_cast<char*>(""));
    std::string label = 
        "{" 
        + std::string("grad: ") + std::to_string(node.grad)
        + std::string("| val: ") + std::to_string(node.value)
        + std::string("| id: ") + std::to_string(node.idx)
        + "|"
        + autodiff::get_str_from_op(node.op)
        + "}";
    agsafeset(n_cur,
        const_cast<char*>("label"),
        const_cast<char*>(label.c_str()),
        "");

    autodiff::Node<T> const & node_left = tape[node.left_child];
    autodiff::Node<T> const & node_right = tape[node.right_child];

    if(node_left.idx != 0) {
        std::string cur_left_id =
            std::to_string(reinterpret_cast<uintptr_t>(&node_left));
        Agnode_t *n_left =
            agnode(g, const_cast<char*>(cur_left_id.c_str()), true);
        agsafeset(n_left,
            const_cast<char*>("label"),
            const_cast<char*>(std::to_string(node_left.value).c_str()),
            "");

        Agedge_t *edge =
            agedge(g, n_left, n_cur, 0, 1);
    }

    if(node_right.idx != 0) {
        std::string cur_right_id =
            std::to_string(reinterpret_cast<uintptr_t>(&node_right));
        Agnode_t *n_right =
            agnode(g, const_cast<char*>(cur_right_id.c_str()), true);
        agsafeset(n_right,
            const_cast<char*>("label"),
            const_cast<char*>(std::to_string(node_right.value).c_str()),
            "");

        Agedge_t *edge =
            agedge(g, n_right, n_cur, 0, 1);
    }

    build_graph(node_left, tape, g);
    build_graph(node_right, tape, g);
}

template <typename T>
void
save_graph_to_file(
    autodiff::Var<T> const & root,
    std::string const & file_name)
{
    GVC_t *gvc = gvContext();
    Agraph_t *g = agopen(const_cast<char*>(""),
        Agdirected, nullptr);

    agsafeset(g,
        const_cast<char*>("rankdir"),
        const_cast<char*>("BT"),
        const_cast<char*>(""));

    autodiff::Tape<T> & tape = root.get_tape_ref();
    build_graph(tape[root.get_node_idx()], tape, g);

    gvLayout(gvc, g, "dot");
    gvRenderFilename(gvc, g, "png", file_name.c_str());

    // Free resources
    gvFreeLayout(gvc, g);
    agclose(g);
    gvFreeContext(gvc);
}

}; //namespace utils
