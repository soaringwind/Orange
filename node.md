百度地图添加自定义的html标签内容
使用的是control方法，需要先设置好control，之后把html的内容全部放到里面，最后在map中加入进去。
        function SelectControl(){
            this.defaultAnchor = BMAP_ANCHOR_TOP_LEFT; 
            this.defaultOffset = new BMap.Size(10, 10);
        }
        SelectControl.prototype = new BMap.Control(); 
        SelectControl.prototype.initialize = function(map){
            var div = document.createElement("div"); 
            var sel = document.createElement("select");
            var but = document.createElement("button");
            var total_opt = document.createElement("option");
            sel.id = "test";
            but.innerText = "确定";
            but.type = "submit";
            but.name = "formbtn"; 
            but.onclick = showSelectHtml;
            total_opt.value = "total";
            total_opt.text = "total";
            sel.add(total_opt, null);
            for (key in info_dict) {
                opt = document.createElement("option");
                opt.value = key;
                opt.text = key;
                sel.add(opt, null);
            }
            div.append(sel);
            div.append(but);
            map.getContainer().appendChild(div);
            return div; 
        }
        var myZoomCtrl = new SelectControl(); 
        map.addControl(myZoomCtrl);
        function showSelectHtml() {
            console.log($("#test option:selected").text());
        }
