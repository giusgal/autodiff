#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <graphviz/gvc.h>
#include <graphviz/cgraph.h>
#include <string>
#include "autodiff.hpp"

namespace utils {

template <typename T>
static void
buildGraph(
    autodiff::Var<T> const & cur,
    Agraph_t *g)
{
    if(cur.getIdx() == 0 ||
        cur.getOperation() == autodiff::Operation::NOP)
    {
        return;
    }

    std::string cur_id =
        std::to_string(reinterpret_cast<uintptr_t>(&cur));
    Agnode_t *n_cur =
        agnode(g, const_cast<char*>(cur_id.c_str()), true);
    agsafeset(n_cur,
        const_cast<char*>("shape"),
        const_cast<char*>("record"),
        const_cast<char*>(""));
    std::string label = 
        "{" + std::to_string(cur.getValue())
        + "|"
        + autodiff::getStrFromOp(cur.getOperation()) + "}";
    agsafeset(n_cur,
        const_cast<char*>("label"),
        const_cast<char*>(label.c_str()),
        "");

    autodiff::Var<T> const & cur_left{
        (autodiff::Tape<T>::getTape())[cur.getLeft()]
    };
    autodiff::Var<T> const & cur_right{
        (autodiff::Tape<T>::getTape())[cur.getRight()]
    };

    if(cur_left.getIdx() != 0) {
        std::string cur_left_id =
            std::to_string(reinterpret_cast<uintptr_t>(&cur_left));
        Agnode_t *n_left =
            agnode(g, const_cast<char*>(cur_left_id.c_str()), true);
        agsafeset(n_left,
            const_cast<char*>("label"),
            const_cast<char*>(std::to_string(cur_left.getValue()).c_str()),
            "");

        Agedge_t *edge =
            agedge(g, n_left, n_cur, 0, 1);
    }

    if(cur_right.getIdx() != 0) {
        std::string cur_right_id =
            std::to_string(reinterpret_cast<uintptr_t>(&cur_right));
        Agnode_t *n_right =
            agnode(g, const_cast<char*>(cur_right_id.c_str()), true);
        agsafeset(n_right,
            const_cast<char*>("label"),
            const_cast<char*>(std::to_string(cur_right.getValue()).c_str()),
            "");

        Agedge_t *edge =
            agedge(g, n_right, n_cur, 0, 1);
    }

    buildGraph(cur_left, g);
    buildGraph(cur_right, g);
}

template <typename T>
void
saveGraphToFile(
    autodiff::Var<T> const &root,
    std::string const &file_name)
{
    GVC_t *gvc = gvContext();
    Agraph_t *g = agopen(const_cast<char*>(""),
        Agdirected, nullptr);

    agsafeset(g,
        const_cast<char*>("rankdir"),
        const_cast<char*>("BT"),
        const_cast<char*>(""));

    buildGraph(root, g);

    gvLayout(gvc, g, "dot");
    gvRenderFilename(gvc, g, "png", file_name.c_str());

    // Free resources
    gvFreeLayout(gvc, g);
    agclose(g);
    gvFreeContext(gvc);
}

}; //namespace utils

#endif // __UTILS_HPP__
