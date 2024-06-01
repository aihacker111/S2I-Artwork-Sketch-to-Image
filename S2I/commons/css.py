css = """
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css');

/* the outermost contrained of the app */
.main{
    display: flex;
    justify-content: center;
    align-items: center;
    width: 1200px;
}

/* #main_row{

} */

/* hide this class */
.svelte-p4aq0j {
    display: none;
}

.wrap.svelte-p4aq0j.svelte-p4aq0j {
    display: none;
}

#download_sketch{
    display: none;
}

#download_output{
    display: none;
}

#column_input, #column_output, #column_segment{
    width: 500px;
    display: flex;
    /* justify-content: center; */
    align-items: center;
}


#tools_header, #segment_header, #input_header, #output_header, #process_header {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 400px;
}


#nn{
    width: 100px;
    height: 100px;
}


#column_process{
    display: flex;
    justify-content: center; /* Center horizontally */
    align-items: center; /* Center vertically */
    height: 600px;
}

/* this is the "pix2pix-turbo" above the process button */
#description > span{
    display: flex;
    justify-content: center; /* Center horizontally */
    align-items: center; /* Center vertically */
}

/* this is the "UNDO_BUTTON, X_BUTTON" */
div.svelte-1030q2h{
    width: 30px;
    height: 30px;
    display: none;
}


#component-5 > div{
    border: 0px;
    box-shadow: none;
}

#cb-eraser, #cb-line{
    display: none;
}

/* eraser text */
#cb-eraser > label > span{
    display: none;
}
#cb-line > label > span{
    display: none;
}


.button-row {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 50px;
    border: 0px;
}
#my-toggle-pencil{
    background-image: url("https://icons.getbootstrap.com/assets/icons/pencil.svg");
    background-color: white;
    background-size: cover;
    margin: 0px;
    box-shadow: none;
    width: 40px;
    height: 40px;
}

#my-toggle-pencil.clicked{
    background-image: url("https://icons.getbootstrap.com/assets/icons/pencil-fill.svg");
    transform: scale(0.98);
    background-color: gray;
    background-size: cover;
    /* background-size: 95%;
    background-position: center; */
    /* border: 2px solid #000; */
    margin: 0px;
    box-shadow: none;
    width: 40px;
    height: 40px;
}


#my-toggle-eraser{
    background-image: url("https://icons.getbootstrap.com/assets/icons/eraser.svg");
    background-color: white;
    background-color: white;
    background-size: cover;
    margin: 0px;
    box-shadow: none;
    width: 40px;
    height: 40px;
}

#my-toggle-eraser.clicked{
    background-image: url("https://icons.getbootstrap.com/assets/icons/eraser-fill.svg");
    transform: scale(0.98);
    background-color: gray;
    background-size: cover;
    margin: 0px;
    box-shadow: none;
    width: 40px;
    height: 40px;
}



#my-button-undo{
    background-image: url("https://icons.getbootstrap.com/assets/icons/arrow-counterclockwise.svg");
    background-color: white;
    background-size: cover;
    margin: 0px;
    box-shadow: none;
    width: 40px;
    height: 40px;
}

#my-button-clear{
    background-image: url("https://icons.getbootstrap.com/assets/icons/x-lg.svg");
    background-color: white;
    background-size: cover;
    margin: 0px;
    box-shadow: none;
    width: 40px;
    height: 40px;

}


#my-button-down{
    background-image: url("https://icons.getbootstrap.com/assets/icons/arrow-down.svg");
    background-color: white;
    background-size: cover;
    margin: 0px;
    box-shadow: none;
    width: 40px;
    height: 40px;

}

.pad2{
    padding: 2px;
    background-color: white;
    border: 2px solid #000;
    margin: 10px;
    display: flex;
    justify-content: center; /* Center horizontally */
    align-items: center; /* Center vertically */
}




#output_image, #input_image, #segment_image{
    border-radius: 0px;
    border: 5px solid #000;
    border-width: none;
}


#output_image > img{
    border: 5px solid #000;
    border-radius: 0px;
    border-width: none;
}

#input_image > div.image-container.svelte-p3y7hu > div.wrap.svelte-yigbas > canvas:nth-child(1){
    border: 5px solid #000;
    border-radius: 0px;
    border-width: none;
}
"""

scripts = """
async () => {
    globalThis.theSketchDownloadFunction = () => {
        console.log("test")
        var link = document.createElement("a");
        dataUri = document.getElementById('download_sketch').href
        link.setAttribute("href", dataUri)
        link.setAttribute("download", "sketch.png")
        document.body.appendChild(link); // Required for Firefox
        link.click();
        document.body.removeChild(link); // Clean up

        // also call the output download function
        theOutputDownloadFunction();
      return false
    }

    globalThis.theOutputDownloadFunction = () => {
        console.log("test output download function")
        var link = document.createElement("a");
        dataUri = document.getElementById('download_output').href
        link.setAttribute("href", dataUri);
        link.setAttribute("download", "output.png");
        document.body.appendChild(link); // Required for Firefox
        link.click();
        document.body.removeChild(link); // Clean up
      return false
    }

    globalThis.UNDO_SKETCH_FUNCTION = () => {
        console.log("undo sketch function")
        var button_undo = document.querySelector('#input_image > div.image-container.svelte-p3y7hu > div.svelte-s6ybro > button:nth-child(1)');
        // Create a new 'click' event
        var event = new MouseEvent('click', {
            'view': window,
            'bubbles': true,
            'cancelable': true
        });
        button_undo.dispatchEvent(event);
    }

    globalThis.DELETE_SKETCH_FUNCTION = () => {
        console.log("delete sketch function")
        var button_del = document.querySelector('#input_image > div.image-container.svelte-p3y7hu > div.svelte-s6ybro > button:nth-child(2)');
        // Create a new 'click' event
        var event = new MouseEvent('click', {
            'view': window,
            'bubbles': true,
            'cancelable': true
        });
        button_del.dispatchEvent(event);
    }

    globalThis.togglePencil = () => {
        el_pencil = document.getElementById('my-toggle-pencil');
        el_pencil.classList.toggle('clicked');
        // simulate a click on the gradio button
        btn_gradio = document.querySelector("#cb-line > label > input");
        var event = new MouseEvent('click', {
            'view': window,
            'bubbles': true,
            'cancelable': true
        });
        btn_gradio.dispatchEvent(event);
        if (el_pencil.classList.contains('clicked')) {
            document.getElementById('my-toggle-eraser').classList.remove('clicked');
            document.getElementById('my-div-pencil').style.backgroundColor = "gray";
            document.getElementById('my-div-eraser').style.backgroundColor = "white";
        }
        else {
            document.getElementById('my-toggle-eraser').classList.add('clicked');
            document.getElementById('my-div-pencil').style.backgroundColor = "white";
            document.getElementById('my-div-eraser').style.backgroundColor = "gray";
        }
    }

    globalThis.toggleEraser = () => {
        element = document.getElementById('my-toggle-eraser');
        element.classList.toggle('clicked');
        // simulate a click on the gradio button
        btn_gradio = document.querySelector("#cb-eraser > label > input");
        var event = new MouseEvent('click', {
            'view': window,
            'bubbles': true,
            'cancelable': true
        });
        btn_gradio.dispatchEvent(event);
        if (element.classList.contains('clicked')) {
            document.getElementById('my-toggle-pencil').classList.remove('clicked');
            document.getElementById('my-div-pencil').style.backgroundColor = "white";
            document.getElementById('my-div-eraser').style.backgroundColor = "gray";
        }
        else {
            document.getElementById('my-toggle-pencil').classList.add('clicked');
            document.getElementById('my-div-pencil').style.backgroundColor = "gray";
            document.getElementById('my-div-eraser').style.backgroundColor = "white";
        }
    }
}
"""