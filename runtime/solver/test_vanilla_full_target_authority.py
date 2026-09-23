from __future__ import annotations
import numpy as np
import production_b14 as prod


def main():
    a_v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=np.float64)
    a_f=np.array([[0,1,2],[0,2,3]],dtype=np.int64)
    b_v=np.array([[2.,0.,0.],[3.,0.,0.],[2.,1.,0.]],dtype=np.float64)
    b_f=np.array([[0,1,2]],dtype=np.int64)
    cache={"slot_pairs":[
        {"slot":"Chest","target_literal_V":a_v,"target_literal_F":a_f},
        {"slot":"Legs","target_literal_V":b_v,"target_literal_F":b_f},
    ]}
    plan=prod._complete_vanilla_target_body_plan(cache,["Chest","Legs"])
    assert plan["enabled"] is False
    assert plan["native_suppression"] == []
    assert len(plan["_collision_triangles"]) == len(a_f)+len(b_f)
    assert all(row["suppressed_target_triangles"] == 0 for row in plan["slots"])
    print("vanilla full-target authority: PASS")

if __name__ == "__main__":
    main()
