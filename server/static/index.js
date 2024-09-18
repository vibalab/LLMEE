window.onload = () =>{


    $('.search-icon').on('click', (e)=>{
        e.preventDefault();
    });
// Select all textareas with the class 'query'
const textareas = $('.query');
const maxHeight = 100; // Set a maximum height for all textareas

function adjustHeight(textarea) {
    // Reset the height to auto to shrink if needed
    textarea.css('height', 'auto');
    
    // Calculate the scroll height
    const scrollHeight = textarea[0].scrollHeight - 50;
    
    // Calculate the new height (min between content height and maxHeight)
    const newHeight = Math.min(scrollHeight, maxHeight);
    textarea.height(newHeight);
    
    // Set overflow based on whether the new height exceeds the max height
    if (newHeight >= maxHeight) {
        textarea.css('overflow-y', 'scroll');
    } else {
        textarea.css('overflow-y', 'hidden');
    }
}

// Attach event listeners for each textarea with the class 'query'
textareas.each(function() {
    const textarea = $(this); // Current textarea element

    textarea.on('input', function() {
        adjustHeight(textarea); // Pass the current textarea to adjustHeight
    });

    textarea.on('keydown', function(e) {
        if (e.key === "Enter") {
            if (e.shiftKey && !e.originalEvent.isComposing) {
                e.preventDefault();
                if (textarea.height() < 150) {                    
                    textarea.css('overflow-y', 'hidden');
                } else {
                    textarea.css('overflow-y', 'scroll');
                }

                const cursorPos = textarea.prop('selectionStart');
                const value = textarea.val();
                const newValue = value.slice(0, cursorPos) + "\n" + value.slice(cursorPos);
                textarea.val(newValue);

                textarea.prop('selectionStart', cursorPos + 1);
                textarea.prop('selectionEnd', cursorPos + 1);
                textarea.scrollTop(textarea[0].scrollHeight);
                adjustHeight(textarea);
            } else if (!e.originalEvent.isComposing) {
                e.preventDefault(); // Optional: Prevent default behavior if needed
                let formData = new FormData();
                formData.append('model_name', $('#model').val()); // Append selected model
                formData.append('dataset', $('#dataset').val()); // Append selected dataset
                formData.append('explanation_method', $('#XAI').val()); // Append selected XAI method
                formData.append('input_prompt', textarea.val()); // Append textarea content

                for (let [key, value] of formData.entries()) {
                    console.log(`${key}: ${value}`);
                }
            }
        }
    });
});




    const modelSelect = $('.model-container');
    
    function fetchModels() {
        $.ajax({
            url: 'https://huggingface.co/api/models?library=transformers&pipeline_tag=text-generation',
            method: 'GET',
            success: function(models) {
                populateModelDropdown(models);
                optionSelect();
            },
            error: function(jqXHR, textStatus, errorThrown) {
                console.error('Error fetching models:', textStatus, errorThrown);
            }
        });
    }

    // Function to populate dropdown with models
    function populateModelDropdown(models) {
        $.each(models, function(index, model) {
            modelSelect.append($('<div>', {
                value: model.modelId,
                text: model.modelId,
                class : 'model-select'
            }));
        });
    }

    // Fetch models when the document is ready
    fetchModels();   


    $('.select-button').on('click', (e) => {
        const isDisplayed = modelSelect.css('height') != '0px';
        if (isDisplayed){
            modelSelect.css('height', '0px');
        }
        else{
            modelSelect.css('height', '20vh');
        }
    })

    const optionSelect = () => $('.model-select').on('click', function(e){
        const selected = $(this).hasClass('selected');

        if(selected){
            $(this).css('background-color', '');
            $(this).removeClass(`selected`);
            $(this).removeAttr('data-model');
            $(`.selected-model[data-model = "${this.innerHTML}"]`).remove();
        }

        else{
            const count = $('.selected-model').length;

            if(count < 3){
                $(this).css('background-color', '#86a7ff');
                $(this).attr('data-model', this.innerHTML);
                $(this).addClass(`selected`);
                const newHtml = `
                <div class="selected-model" data-model = "${this.innerHTML}">
                    ${this.innerHTML}
                    <span class="remove-button">&times;</span>
                </div>
                `;

                $('#selected-models').append(newHtml);

                $('#selected-models .remove-button').on('click', function() {
                    const model = $(this).parent().data('model');
                    const dropItem = $(`.selected[data-model = "${model}"]`);
                    dropItem.remove();
                    $(this).parent().remove();
                    dropItem.css('background-color', 'white');
                    dropItem.removeClass(`selected`);
                    dropItem.removeData('model');
                });

            }

            else{
                alert('you already selected 3 models');
            }
        }

        const count = $('.selected-model').length;

        if(count > 0){
            $('#selected-models').css('display', 'block');
        } else {
            $('#selected-models').css('display', 'none');
        }
    })

    $('.model-input').on('keydown', function(e){
        if(e.key == "Enter"){
            e.preventDefault();
            let count = $('.selected-model').length;

                if(count < 3){
                    const newHtml = `
                    <div class="selected-model" data-model = "${$(this).val()}">
                        ${$(this).val()}
                        <span class="remove-button">&times;</span>
                    </div>
                    `;

                    $('#selected-models').append(newHtml);

                    $('#selected-models .remove-button').on('click', function() {
                        const model = $(this).parent().data('model');
                        $(this).parent().remove();
                    });

                }

                else{
                    alert('you already selected 3 models');
                }

                $(this).val("");

                count = $('.selected-model').length;

                if(count > 0){
                    $('#selected-models').css('display', 'block');
                } else {
                    $('#selected-models').css('display', 'none');
                }
            }


    })


    $('.selected-xai').on('click', function(e){
        const isDisplayed = $('.xai-list').css('max-height') != '0px';
        if (isDisplayed){
            $('.xai-list').css('max-height', '0px');
        } else{
            $('.xai-list').css('max-height', '1000px');
        }
    })

    $('.xai-item').on('click', function(e){
        $('.selected-xai').html($(this).html());
        $('.selected-xai').click();
    })

    $(document).on('click', function(e) {
        const modelSelect = $('.select-button');
        const modelContainer = $('.model-container');
        if (!modelSelect.is(e.target) && modelSelect.has(e.target).length === 0 &&
        !modelContainer.is(e.target) && modelContainer.has(e.target).length === 0) {
            modelContainer.css('height', '0px'); // Hide the dropdown
        }


        const xaiSelect = $('.selected-xai');
        const xaiList = $('.xai-list');
        if (!xaiSelect.is(e.target) && xaiSelect.has(e.target).length === 0 &&
            !xaiList.is(e.target) && xaiList.has(e.target).length === 0) {
                xaiList.css('max-height', '0px'); // Hide the dropdown
        }
    });


    $('#load-button').on('click', function(e){
        e.preventDefault();

        let formData = new FormData();
        const model_list = $('.selected-model').map(function(){return $(this).data('model')}).get();
        formData.append('model_list', JSON.stringify(model_list));

        // console.log(model_list);

        fetch('/loadModel',{
            method : 'POST',
            body : formData,
        }).then((res) => res.json())
        .then((data)=>{
            console.log(data);
        })
    })


    //ToDo(YSKim): print model info
    $('#enter').on('click', function(e){
        e.preventDefault();

        let formData = new FormData();
        formData.append('input', $('.query')[0].value);
        formData.append('output', $('.query')[1].value);
        formData.append('xai-method', $('.selected-xai').html());
        const model_list = $('.selected-model').map(function(){return $(this).data('model')}).get();
        formData.append('model_list', JSON.stringify(model_list));

        $('.answer-area').html("")
        $('.answer-area').append(`Predicted Output: ${$('.query')[1].value}`);

        // Make the POST request to '/input' and handle the response
        fetch('/input',{
            method : 'POST',
            body : formData,
        }).then((res) => res.json())
        .then((data)=>{
            console.log(data);

            model_list.forEach(function(model){
                if (data.results && data.results[model]) {
                    const seq_attr = data.results[model]['seq_attr']; // Get the seq_attr list
                
                    // Split the input sentence into words
                    const inputSentence = $('.query')[0].value;
                    // console.log(seq_attr.length)
                    // console.log(inputSentence.length)
                     const card = $('<div>', {
                        class : `model-card ${model}`
                    });

                    const cardHeader = $('<div>', {
                        text: model,
                        class : `card-header ${model}`
                    });

                    const cardBody = $('<div>', {
                        class : `card-body ${model}`
                    });

                    cardBody.html(`
                        <p>Input     Text: ${inputSentence} </p>
                        <p>Generated Text: ${data.results[model]['generated_text']}</p>
                        <p>ROGUE    Score: 0.xxx</p> 
                        <p>???      Score: 0.xxx</p>
                        `);



                    const metaData = data.results[model]['model_card'];
                    const content = $('<table>', {
                        class:`card-meta ${model}`
                    });
                    let row;
            
                    Object.entries(metaData).forEach(([key, value], index) => {
                        // replace null to 'N/A'
                        if (value === null || (Array.isArray(value) && value.length === 0)) {
                            value = 'N/A';
                        } else if (Array.isArray(value)) {
                            value = value.join(', ');
                        }
            
                        key = key.replace(/_/g, ' ').toLowerCase(); // replace '_' to ' '
                        key = key.charAt(0).toUpperCase() + key.slice(1);  //make first char uppercase
                        let cell = `<td>${key}: ${value}</td>`;
            
                        if (index % 2 === 0) {  
                            row = $('<tr>');
                            content.append(row);
                        }
                        row.append(cell);  
                    });
            
                    cardBody.append(content);  
                    // let colorCodedSentence = '';

                    // // Loop through each word and corresponding seq_attr value
                    // inputWords.forEach((word, index) => {
                    //     let attrValue = seq_attr[index];
                    //     let color = getColorFromSeqAttr(attrValue); // Function to determine color based on the seq_attr value
                    //     colorCodedSentence += `<span style="color: ${color}">${word}</span> `;
                    // });
                    
                    // cardBody.html(`<p>${colorCodedSentence}</p>`);

                    card.append(cardHeader);
                    card.append(cardBody);


                    $('.answer-area').append(card);
                }
            })
            headerEvent();

            console.log(formData);
        })
    })

    function headerEvent(){
        $('.card-header').on('click', function(e){
            const isShirinked = $(this).parent().css('max-height') == "100px";
    
            console.log(isShirinked);
            
            if(isShirinked){
                // $(this).parent().css('flex-basis', "");
                $(this).parent().css('max-height', "5000px");
                // $(this).next().css('overflow-y', "scroll");
                // $(this).next().css('height', "auto");
            } else{
                // $(this).parent().css('flex-basis', "100px");
                $(this).parent().css('max-height', "100px");
                // $(this).next().css('overflow-y', "hidden");
                // $(this).next().css('height', "100px");
            }        
        })
    }

    // // Function to get the color based on seq_attr value
    // function getColorFromSeqAttr(value) {
    //     if (value > 0) {
    //         // Map positive values to blue scale (e.g., light blue to dark blue)
    //         let blueIntensity = Math.min(255, Math.floor(255 * value)); // Scale the value for color intensity
    //         return `rgb(0, 0, ${blueIntensity})`;
    //     } else {
    //         // Map negative values to red scale (e.g., light red to dark red)
    //         let redIntensity = Math.min(255, Math.floor(255 * Math.abs(value))); // Scale the value for color intensity
    //         return `rgb(${redIntensity}, 0, 0)`;
    //     }
    // }
}