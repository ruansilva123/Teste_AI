$(function(){
    const uploadInput = $("#input_file");
    const submitButton = $("#submit");
    const output = $(".output");
    
    output.text("sem status");
    
    submitButton.on("click", async function(event){
        event.preventDefault();
    
        let formData = new FormData();
        formData.append('file', uploadInput[0].files[0]);
    

        let data = await get_data(formData, output)
        console.log(data)
    //     $.post({
    //         url: 'http://127.0.0.1:8000/upload-image/',
    //         type: 'POST',
    //         data: formData,
    //         processData: false,
    //         contentType: false,
    //     })
    //     .done(async function(response) {
    //         console.log(response);
    //         output.text("Upload bem-sucedido!");
    //     })
    //     .fail(async function(error) {
    //         console.log(error);
    //         output.text("Erro no upload.");
    //     })
    // });

    })
})


async function get_data(data, output){
    return new Promise((resolve,reject)=>{
        $.post({
            url: 'http://127.0.0.1:8000/upload-image/',
            data: data,
            processData: false,
            contentType: false,
        })
        .done((response)=>{
            resolve({
                "data":response
            });
            let confidence = parseFloat(response.confidence.toFixed(2));
            output.text(`${confidence}% \n ${response.predicted_class}`);
        })
        .fail((error)=>{
            reject({"data":error})
        });
        output.text("Erro no upload.");
    })
}