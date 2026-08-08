"""
Generate self-contained HTML for evolution commit graph visualization.

Uses embedded D3.js v7 for SVG rendering. The output HTML is fully
offline-capable — no external dependencies.

Supports three node types:
- iteration_final: large circles (main iteration commits)
- eval_snapshot: small circles (evaluate() snapshots within iterations)
- meta_evolve: diamonds (meta-evolution commits)
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# HTML template — single self-contained file
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evolution Commit Graph</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#0d1117;color:#c9d1d9;overflow:hidden;height:100vh}
.header{background:#161b22;border-bottom:1px solid #30363d;padding:12px 20px;display:flex;align-items:center;gap:20px;flex-wrap:wrap}
.header h1{font-size:16px;color:#58a6ff;font-weight:600}
.header .stat{font-size:13px;color:#8b949e}
.header .stat b{color:#c9d1d9}
.main{display:flex;height:calc(100vh - 49px)}
.graph-area{flex:1;position:relative;overflow:hidden;background:#0d1117}
.graph-area svg{width:100%;height:100%}
.sidebar{width:340px;background:#161b22;border-left:1px solid #30363d;display:flex;flex-direction:column;overflow:hidden}
.sidebar-section{padding:12px 16px;border-bottom:1px solid #21262d}
.sidebar-section h3{font-size:12px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px}
.reward-chart{height:180px}
.detail-panel{flex:1;overflow-y:auto;padding:16px;font-size:13px}
.detail-panel .field{margin-bottom:10px}
.detail-panel .field-label{font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;margin-bottom:2px}
.detail-panel .field-value{color:#c9d1d9;word-break:break-all}
.detail-panel .file-list{list-style:none;padding:0}
.detail-panel .file-list li{padding:3px 0;display:flex;justify-content:space-between;font-size:12px}
.detail-panel .file-list .added{color:#3fb950}
.detail-panel .file-list .removed{color:#f85149}
.empty-detail{color:#484f58;text-align:center;padding:40px 20px;font-size:13px}
.tooltip{position:absolute;background:#1c2128;border:1px solid #30363d;border-radius:6px;padding:8px 12px;font-size:12px;pointer-events:none;z-index:100;max-width:300px;box-shadow:0 4px 12px rgba(0,0,0,.4)}
.tooltip .tt-type{color:#a371f7;font-weight:600;font-size:10px;text-transform:uppercase;letter-spacing:.5px}
.tooltip .tt-iter{color:#58a6ff;font-weight:600}
.tooltip .tt-reward{color:#e3b341}
.tooltip .tt-summary{color:#8b949e;margin-top:4px;font-size:11px}
.controls{position:absolute;bottom:16px;left:16px;display:flex;gap:8px}
.controls button{background:#21262d;border:1px solid #30363d;color:#c9d1d9;padding:6px 12px;border-radius:6px;cursor:pointer;font-size:12px}
.controls button:hover{background:#30363d}
.sparkline{display:inline-block;vertical-align:middle}
.node-best{filter:url(#glow)}
.iter-label{font-size:12px;fill:#484f58;font-weight:600}
</style>
</head>
<body>
<div class="header">
  <h1>Evolution Graph</h1>
  <span class="stat" id="stat-goal"></span>
  <span class="stat">Iterations: <b id="stat-iters">0</b></span>
  <span class="stat">Best Reward: <b id="stat-best">—</b></span>
  <span class="stat" id="stat-time"></span>
</div>
<div class="main">
  <div class="graph-area">
    <svg id="graph-svg"></svg>
    <div class="controls">
      <button onclick="resetZoom()">Reset View</button>
    </div>
  </div>
  <div class="sidebar">
    <div class="sidebar-section reward-chart">
      <h3>Reward Trend</h3>
      <svg id="reward-svg" width="308" height="150"></svg>
    </div>
    <div class="sidebar-section" style="flex:1;overflow:hidden;display:flex;flex-direction:column">
      <h3>Node Detail</h3>
      <div class="detail-panel" id="detail-panel">
        <div class="empty-detail">Click a node to view details</div>
      </div>
    </div>
  </div>
</div>
<div class="tooltip" id="tooltip" style="display:none"></div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
var DATA = __DATA_PLACEHOLDER__;
</script>
<script>
(function(){
"use strict";

/* ---- helpers ---- */
function rewardColor(r, minR, maxR){
  if(r==null) return "#484f58";
  var t = maxR===minR ? 0.5 : (r-minR)/(maxR-minR);
  var h = t * 120;
  return "hsl("+h+",80%,50%)";
}

function typeLabel(t){
  if(t==="iteration_final") return "Final Commit";
  if(t==="eval_snapshot") return "Eval Snapshot";
  if(t==="meta_evolve") return "Meta-Evolve";
  return t;
}

function renderDiffStats(n){
  if(!n.diff_stats || !n.diff_stats.files || !n.diff_stats.files.length) return "";
  var h='<div class="field"><div class="field-label">Files Changed ('+n.diff_stats.files_changed+' | +'+n.diff_stats.insertions+' / -'+n.diff_stats.deletions+')</div>';
  h+='<ul class="file-list">';
  n.diff_stats.files.forEach(function(f){
    h+='<li><span>'+f.path+'</span><span><span class="added">+'+f.added+'</span> <span class="removed">-'+f.removed+'</span></span></li>';
  });
  h+='</ul></div>';
  return h;
}

/* ---- header stats ---- */
var meta = DATA.meta;
document.getElementById("stat-iters").textContent = meta.total_iterations;
if(meta.best_version && meta.best_version.reward!=null){
  document.getElementById("stat-best").textContent = meta.best_version.reward.toFixed(4);
  if(meta.best_version.commit){
    document.getElementById("stat-best").textContent += " ("+meta.best_version.commit+")";
  }
}
if(meta.generated_at){
  document.getElementById("stat-time").textContent = "Generated: "+meta.generated_at.replace("T"," ").slice(0,19);
}
if(meta.goal){
  var goalEl = document.getElementById("stat-goal");
  goalEl.textContent = meta.goal.length>80 ? meta.goal.slice(0,80)+"..." : meta.goal;
  goalEl.title = meta.goal;
}

var allNodes = DATA.nodes;
var edges = DATA.edges;
var iterGroups = DATA.iteration_groups || [];
if(!allNodes.length) return;

/* ---- classify nodes ---- */
var finalNodes = allNodes.filter(function(n){return n.node_type==="iteration_final"});
var snapNodes = allNodes.filter(function(n){return n.node_type==="eval_snapshot"});
var metaNodes = allNodes.filter(function(n){return n.node_type==="meta_evolve"});

/* ---- reward range from final nodes only ---- */
var rewards = finalNodes.map(function(n){return n.reward}).filter(function(r){return r!=null});
var minR = rewards.length ? Math.min.apply(null,rewards) : 0;
var maxR = rewards.length ? Math.max.apply(null,rewards) : 1;
var maxActions = Math.max.apply(null, finalNodes.map(function(n){return n.action_count||1}));

/* ---- build id->node map ---- */
var nodeMap = {};
allNodes.forEach(function(n){nodeMap[n.id]=n});

/* ---- layout ---- */
var yStep = 120;
var snapYOffset = -40;   // snapshots above final
var metaYOffset = 45;    // meta below final
var snapXSpacing = 28;
var metaXOffset = 80;

// Group final nodes by iteration
var iterMap = {};
finalNodes.forEach(function(n){
  iterMap[n.iteration] = n;
});

// Position iteration_final nodes
finalNodes.forEach(function(n){
  n._y = -n.iteration * yStep;
  n._x = 0;
});

// Position eval_snapshot nodes
var snapsByIter = {};
snapNodes.forEach(function(n){
  if(!snapsByIter[n.iteration]) snapsByIter[n.iteration]=[];
  snapsByIter[n.iteration].push(n);
});
Object.keys(snapsByIter).forEach(function(iter){
  var snaps = snapsByIter[iter];
  var finalNode = iterMap[parseInt(iter)];
  if(!finalNode) return;
  var baseX = finalNode._x;
  var baseY = finalNode._y;
  var totalWidth = (snaps.length-1)*snapXSpacing;
  snaps.forEach(function(s, i){
    s._x = baseX - totalWidth/2 + i*snapXSpacing;
    s._y = baseY + snapYOffset;
  });
});

// Position meta_evolve nodes
var metaByIter = {};
metaNodes.forEach(function(n){
  var mainIter = n.meta_main_iteration != null ? n.meta_main_iteration : n.iteration;
  metaByIter[mainIter] = n;
});
metaNodes.forEach(function(n){
  var mainIter = n.meta_main_iteration != null ? n.meta_main_iteration : n.iteration;
  var finalNode = iterMap[mainIter];
  if(!finalNode) return;
  n._x = finalNode._x + metaXOffset;
  n._y = finalNode._y + metaYOffset;
});

/* ---- main graph SVG ---- */
var svg = d3.select("#graph-svg");
var g = svg.append("g");

// defs for glow + diamond marker
var defs = svg.append("defs");
var filter = defs.append("filter").attr("id","glow");
filter.append("feGaussianBlur").attr("stdDeviation","3").attr("result","coloredBlur");
var feMerge = filter.append("feMerge");
feMerge.append("feMergeNode").attr("in","coloredBlur");
feMerge.append("feMergeNode").attr("in","SourceGraphic");

// zoom
var zoomBehavior = d3.zoom().scaleExtent([0.2,5]).on("zoom",function(event){
  g.attr("transform",event.transform);
});
svg.call(zoomBehavior);
window.resetZoom = function(){
  var svgEl = document.getElementById("graph-svg");
  var w = svgEl.clientWidth, h = svgEl.clientHeight;
  var minY = 0, maxY = 0;
  allNodes.forEach(function(n){if(n._y<minY)minY=n._y;if(n._y>maxY)maxY=n._y});
  var midY = (minY+maxY)/2;
  var scale = Math.min(w/400, h/(maxY-minY+200));
  scale = Math.min(scale, 1.5);
  svg.transition().duration(500).call(
    zoomBehavior.transform,
    d3.zoomIdentity.translate(w/2, h/2-midY*scale).scale(scale)
  );
};

/* ---- background bands ---- */
var bandColors = ["#161b2280","#0d111780"];
finalNodes.forEach(function(n, idx){
  g.append("rect")
    .attr("x", -300)
    .attr("y", n._y + snapYOffset - 15)
    .attr("width", 600)
    .attr("height", yStep - 10)
    .attr("fill", bandColors[idx % 2])
    .attr("rx", 4);
});

/* ---- iteration labels ---- */
finalNodes.forEach(function(n){
  g.append("text")
    .attr("class","iter-label")
    .attr("x", -140)
    .attr("y", n._y + 5)
    .attr("text-anchor","end")
    .text("Round "+n.iteration);
});

/* ---- edges ---- */
var edgeG = g.selectAll(".edge").data(edges).enter().append("line")
  .attr("class","edge")
  .attr("x1",function(e){return nodeMap[e.source]?nodeMap[e.source]._x:0})
  .attr("y1",function(e){return nodeMap[e.source]?nodeMap[e.source]._y:0})
  .attr("x2",function(e){return nodeMap[e.target]?nodeMap[e.target]._x:0})
  .attr("y2",function(e){return nodeMap[e.target]?nodeMap[e.target]._y:0})
  .attr("stroke",function(e){
    if(e.type==="crossover") return "#a371f7";
    if(e.type==="archive_switch") return "#d29922";
    if(e.type==="meta_evolve_bridge") return "#a371f7";
    if(e.type==="within_iteration") return "#58a6ff";
    return "#30363d";
  })
  .attr("stroke-width",function(e){
    if(e.type==="within_iteration") return 1.5;
    if(e.type==="refine") return 2;
    return 2;
  })
  .attr("stroke-dasharray",function(e){
    if(e.type==="meta_evolve_bridge") return "6,3";
    if(e.type==="archive_switch") return "5,3";
    if(e.type==="crossover") return "5,3";
    return "none";
  });

/* ---- eval_snapshot nodes ---- */
var snapG = g.selectAll(".snap-node").data(snapNodes).enter().append("g")
  .attr("class","snap-node")
  .attr("transform",function(n){return "translate("+n._x+","+n._y+")"})
  .style("cursor","pointer");

snapG.append("circle")
  .attr("r",function(n){
    if(n.is_selected) return 7;
    return 5;
  })
  .attr("fill",function(n){return rewardColor(n.reward,minR,maxR)})
  .attr("stroke",function(n){
    if(n.is_selected) return "#ffd700";
    return "#30363d";
  })
  .attr("stroke-width",function(n){
    if(n.is_selected) return 2.5;
    return 1;
  });

// eval_mode badge for selected snapshots
snapG.filter(function(n){return n.is_selected})
  .append("text")
  .attr("y", -10)
  .attr("text-anchor","middle")
  .attr("fill","#ffd700")
  .attr("font-size","9px")
  .attr("font-weight","600")
  .text(function(n){return n.eval_mode || "selected"});

/* ---- iteration_final nodes ---- */
var finalG = g.selectAll(".final-node").data(finalNodes).enter().append("g")
  .attr("class",function(n){return "final-node"+(n.is_best?" node-best":"")})
  .attr("transform",function(n){return "translate("+n._x+","+n._y+")"})
  .style("cursor","pointer");

finalG.append("circle")
  .attr("r",function(n){return 8+8*(n.action_count||1)/maxActions})
  .attr("fill",function(n){return rewardColor(n.reward,minR,maxR)})
  .attr("stroke",function(n){return n.is_best?"#ffd700":"#30363d"})
  .attr("stroke-width",function(n){return n.is_best?3:1.5});

finalG.append("text")
  .attr("y",function(n){return -(8+8*(n.action_count||1)/maxActions)-6})
  .attr("text-anchor","middle")
  .attr("fill","#58a6ff")
  .attr("font-size","10px")
  .text(function(n){return n.reward!=null?n.reward.toFixed(3):""});

/* ---- meta_evolve nodes (diamond shape) ---- */
var metaG = g.selectAll(".meta-node").data(metaNodes).enter().append("g")
  .attr("class","meta-node")
  .attr("transform",function(n){return "translate("+n._x+","+n._y+")"})
  .style("cursor","pointer")
  .style("opacity",function(n){return n.is_noop?0.45:1});

metaG.append("polygon")
  .attr("points","0,-14 14,0 0,14 -14,0")
  .attr("fill","#a371f7")
  .attr("stroke","#30363d")
  .attr("stroke-width",1.5);

metaG.append("text")
  .attr("text-anchor","middle")
  .attr("fill","white")
  .attr("font-size","10px")
  .attr("font-weight","bold")
  .attr("dy","0.35em")
  .text("M");

/* ---- tooltip ---- */
var tooltip = document.getElementById("tooltip");
var allNodeGs = [];
finalG.each(function(n){allNodeGs.push({el:this,node:n})});
snapG.each(function(n){allNodeGs.push({el:this,node:n})});
metaG.each(function(n){allNodeGs.push({el:this,node:n})});

allNodeGs.forEach(function(item){
  d3.select(item.el).on("mouseenter",function(event,n){
    tooltip.style.display="block";
    var html='<div class="tt-type">'+typeLabel(n.node_type)+'</div>';
    html+='<div class="tt-iter">Iteration '+n.iteration+'</div>';
    html+='<div class="tt-reward">Reward: '+(n.reward!=null?n.reward.toFixed(4):"N/A")+'</div>';
    if(n.node_type==="eval_snapshot"){
      html+='<div class="tt-summary">eval_mode: '+(n.eval_mode||"—")+(n.is_selected?" ★ SELECTED":"")+'</div>';
    }
    if(n.node_type==="meta_evolve"){
      html+='<div class="tt-summary">Main iter: '+n.meta_main_iteration+'</div>';
    }
    var summary = n.node_type==="meta_evolve" ? n.meta_summary : n.summary_text;
    if(summary){
      html+='<div class="tt-summary">'+summary.slice(0,200)+'</div>';
    }
    tooltip.innerHTML=html;
  })
  .on("mousemove",function(event){
    tooltip.style.left=(event.offsetX+12)+"px";
    tooltip.style.top=(event.offsetY-10)+"px";
  })
  .on("mouseleave",function(){
    tooltip.style.display="none";
  });
});

/* ---- click detail ---- */
var selectedShape = null;
var selectedStroke = null;

function handleClick(event,n){
  if(selectedShape){
    d3.select(selectedShape).attr("stroke-width",selectedStroke);
  }
  var shape = this.querySelector("circle") || this.querySelector("polygon");
  selectedShape = shape;
  selectedStroke = shape ? shape.getAttribute("stroke-width") : null;
  if(shape) d3.select(shape).attr("stroke-width",3.5);

  var panel = document.getElementById("detail-panel");
  var html = '';

  if(n.node_type==="eval_snapshot"){
    html += '<div class="field"><div class="field-label">Type</div><div class="field-value">Eval Snapshot</div></div>';
    html += '<div class="field"><div class="field-label">Iteration</div><div class="field-value">'+n.iteration+'</div></div>';
    html += '<div class="field"><div class="field-label">Snapshot #</div><div class="field-value">'+n.snapshot_index+'</div></div>';
    html += '<div class="field"><div class="field-label">Reward</div><div class="field-value">'+(n.reward!=null?n.reward.toFixed(4):"N/A")+'</div></div>';
    html += '<div class="field"><div class="field-label">Eval Mode</div><div class="field-value">'+(n.eval_mode||"—")+'</div></div>';
    html += '<div class="field"><div class="field-label">Code Hash</div><div class="field-value"><code>'+(n.code_hash||"—")+'</code></div></div>';
    html += '<div class="field"><div class="field-label">Selected</div><div class="field-value">'+(n.is_selected?"Yes ★":"No")+'</div></div>';
  } else if(n.node_type==="meta_evolve"){
    html += '<div class="field"><div class="field-label">Type</div><div class="field-value" style="color:#a371f7;font-weight:600">Meta-Evolve</div></div>';
    html += '<div class="field"><div class="field-label">Main Iteration</div><div class="field-value">'+n.meta_main_iteration+'</div></div>';
    html += '<div class="field"><div class="field-label">Commit</div><div class="field-value"><code>'+(n.full_hash||"").slice(0,12)+'</code></div></div>';
    if(n.timestamp) html += '<div class="field"><div class="field-label">Time</div><div class="field-value">'+n.timestamp+'</div></div>';
    if(n.commit_message) html += '<div class="field"><div class="field-label">Message</div><div class="field-value">'+n.commit_message+'</div></div>';
    html += '<div class="field"><div class="field-label">Modifications</div><div class="field-value">'+(n.meta_modifications_count||0)+'</div></div>';
    if(n.meta_summary) html += '<div class="field"><div class="field-label">Summary</div><div class="field-value">'+n.meta_summary+'</div></div>';
    html += renderDiffStats(n);
  } else {
    // iteration_final
    html += '<div class="field"><div class="field-label">Type</div><div class="field-value">Final Commit</div></div>';
    html += '<div class="field"><div class="field-label">Iteration</div><div class="field-value">'+n.iteration+'</div></div>';
    html += '<div class="field"><div class="field-label">Reward</div><div class="field-value">'+(n.reward!=null?n.reward.toFixed(4):"N/A")+'</div></div>';
    html += '<div class="field"><div class="field-label">Commit</div><div class="field-value"><code>'+n.full_hash.slice(0,12)+'</code></div></div>';
    if(n.is_best) html += '<div class="field"><div class="field-label">Status</div><div class="field-value" style="color:#ffd700;font-weight:600">★ Best Version</div></div>';
    if(n.selected_snapshot_index!=null) html += '<div class="field"><div class="field-label">Selected Snapshot</div><div class="field-value">#'+n.selected_snapshot_index+'</div></div>';
    if(n.timestamp) html += '<div class="field"><div class="field-label">Time</div><div class="field-value">'+n.timestamp+'</div></div>';
    if(n.commit_message) html += '<div class="field"><div class="field-label">Message</div><div class="field-value">'+n.commit_message+'</div></div>';
    if(n.end_reason) html += '<div class="field"><div class="field-label">End Reason</div><div class="field-value">'+n.end_reason+'</div></div>';
    if(n.summary_text) html += '<div class="field"><div class="field-label">Summary</div><div class="field-value">'+n.summary_text+'</div></div>';
    if(n.action_count) html += '<div class="field"><div class="field-label">Actions</div><div class="field-value">'+n.action_count+'</div></div>';

    html += renderDiffStats(n);

    if(n.reward_history && n.reward_history.length){
      html += '<div class="field"><div class="field-label">In-iteration Reward History ('+n.reward_history.length+' evals)</div>';
      html += '<svg class="sparkline" width="260" height="40" style="display:block;margin-top:4px">';
      var rh = n.reward_history.map(function(h){return h.reward!=null?h.reward:0});
      var rhMin = Math.min.apply(null,rh), rhMax = Math.max.apply(null,rh);
      var rhW = 260, rhH = 36, pad = 2;
      var pts = rh.map(function(v,i){
        var x = pad + i/(rh.length-1||1)*(rhW-2*pad);
        var y = pad + (1-(v-rhMin)/(rhMax-rhMin||1))*(rhH-2*pad);
        return x+","+y;
      });
      html += '<polyline points="'+pts.join(" ")+'" fill="none" stroke="#58a6ff" stroke-width="1.5"/>';
      rh.forEach(function(v,i){
        var x = pad + i/(rh.length-1||1)*(rhW-2*pad);
        var y = pad + (1-(v-rhMin)/(rhMax-rhMin||1))*(rhH-2*pad);
        var mode = n.reward_history[i].eval_mode||"";
        var isSelected = n.selected_snapshot_index===i;
        var fill = mode==="val"?"#a371f7":"#58a6ff";
        if(isSelected) fill="#ffd700";
        html += '<circle cx="'+x+'" cy="'+y+'" r="'+(isSelected?4:2.5)+'" fill="'+fill+'" '+(isSelected?'stroke="#fff" stroke-width="1"':'')+'>';
        html += '<title>'+(v.toFixed?v.toFixed(4):v)+' ('+mode+')'+(isSelected?' ★ SELECTED':'')+'</title></circle>';
      });
      html += '</svg></div>';
    }
  }

  panel.innerHTML = html;
}

finalG.on("click", handleClick);
snapG.on("click", handleClick);
metaG.on("click", handleClick);

/* ---- reward trend chart (final nodes only) ---- */
(function(){
  var rsvg = d3.select("#reward-svg");
  var margin = {top:10,right:16,bottom:24,left:40};
  var width = 308-margin.left-margin.right;
  var height = 150-margin.top-margin.bottom;

  var g2 = rsvg.append("g").attr("transform","translate("+margin.left+","+margin.top+")");

  var sorted = finalNodes.slice().sort(function(a,b){return a.iteration-b.iteration});
  if(!sorted.length) return;
  var x = d3.scaleLinear().domain(d3.extent(sorted,function(n){return n.iteration})).range([0,width]);
  var y = d3.scaleLinear().domain([minR,maxR]).range([height,0]).nice();

  // axes
  g2.append("g").attr("transform","translate(0,"+height+")").call(d3.axisBottom(x).ticks(Math.min(sorted.length,10)).tickFormat(d3.format("d"))).selectAll("text,line,path").attr("stroke","#484f58").attr("fill","#484f58");
  g2.append("g").call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".2f"))).selectAll("text,line,path").attr("stroke","#484f58").attr("fill","#484f58");

  // line
  var line = d3.line().x(function(n){return x(n.iteration)}).y(function(n){return y(n.reward)}).defined(function(n){return n.reward!=null});
  g2.append("path").datum(sorted).attr("fill","none").attr("stroke","#58a6ff").attr("stroke-width",2).attr("d",line);

  // dots
  var dots = g2.selectAll(".dot").data(sorted).enter().append("circle")
    .attr("cx",function(n){return x(n.iteration)})
    .attr("cy",function(n){return y(n.reward)})
    .attr("r",function(n){return n.is_best?5:3.5})
    .attr("fill",function(n){return rewardColor(n.reward,minR,maxR)})
    .attr("stroke",function(n){return n.is_best?"#ffd700":"none"})
    .attr("stroke-width",function(n){return n.is_best?2:0})
    .style("cursor","pointer");

  dots.on("click",function(event,n){
    finalG.each(function(d){
      if(d.id===n.id){
        d3.select(this.querySelector("circle")).dispatch("click");
      }
    });
  });

  dots.append("title").text(function(n){return "iter "+n.iteration+": "+(n.reward!=null?n.reward.toFixed(4):"N/A")});
})();

/* ---- auto-fit ---- */
setTimeout(resetZoom, 100);

})();
</script>
</body>
</html>"""


def _get_d3_inline() -> Optional[str]:
    """Return D3.js v7 minified as a string for embedding.

    Falls back to None (CDN will be used) if local copy is unavailable.
    """
    try:
        return (Path(__file__).parent / "d3.v7.min.js").read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None


def render_evolution_html(data: Dict[str, Any]) -> str:
    """Render visualization data as a self-contained HTML string.

    Tries to embed D3.js inline for offline use; falls back to CDN.
    """
    data_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    d3_inline = _get_d3_inline()
    html = _HTML_TEMPLATE

    if d3_inline:
        html = html.replace(
            '<script src="https://d3js.org/d3.v7.min.js"></script>',
            f"<script>\n{d3_inline}\n</script>",
        )
    html = html.replace("__DATA_PLACEHOLDER__", data_json)
    return html
