window.onload = () =>{
    $('.search-icon').on('click', (e)=>{
        e.preventDefault();
        console.log('icon clicked');
    });

    const textarea = $('.query');
    const maxHeight = 100;

    function adjustHeight() {
        // Reset the height to auto to shrink if needed
        // textarea.css('height', 'auto');
        const text = textarea.val();
        const numberOfLines = (text.match(/\n/g) || []).length + 1;
        const newHeight = Math.min(numberOfLines * 15, maxHeight);
        textarea.height(newHeight);

        // Set overflow based on whether the new height exceeds the max height
        if (newHeight >= maxHeight) {
            textarea.css('overflow-y', 'scroll');
        } else {
            textarea.css('overflow-y', 'hidden');
        }
    }

    // $('#query').on('keydown', (e)=>{
    //     if e.key == 'ENTER'
    // });

    $('.query').on('keydown', function(e) {
        if (e.key === "Enter") {
            if (e.shiftKey && !e.originalEvent.isComposing) {
                e.preventDefault(); // Optional: Prevent default behavior if needed
                // if (textarea.height() < 150) {                    
                //     textarea.height(textarea.height() + 15);
                //     textarea.css('overflow-y', 'hidden');
                // } else {
                //     textarea.css('overflow-y', 'scroll');
                // }

                const cursorPos = textarea.prop('selectionStart');
                // const value = textarea.val();
                // const newValue = value.substring(0, cursorPos) + "\n";
                // textarea.val(newValue);


                textarea[0].setRangeText('\n', cursorPos, cursorPos, 'end')
                textarea.prop('selectionStart', cursorPos + 1);
                textarea.prop('selectionEnd', cursorPos + 1);
                textarea.scrollTop(textarea[0].scrollHeight);
                adjustHeight();
            } else if (!e.originalEvent.isComposing){
                // If only Enter is pressed
                e.preventDefault(); // Optional: Prevent default behavior if needed
                let formData = new FormData();
                formData.append('model_name', $('#model').val()); // Append selected model
                formData.append('dataset', $('#dataset').val()); // Append selected dataset
                formData.append('explanation_method', $('#XAI').val()); // Append selected XAI method
                formData.append('input_prompt', $('#query').val()); // Append textarea content

                for (let [key, value] of formData.entries()) {
                    console.log(`${key}: ${value}`);
                }
            }
        }
    });

    textarea.on('input', function() {
        adjustHeight();
    });



    const modelSelect = $('#model');
    
    function fetchModels() {
        $.ajax({
            url: 'https://huggingface.co/api/models?library=transformers',
            method: 'GET',
            success: function(models) {
                populateModelDropdown(models);
            },
            error: function(jqXHR, textStatus, errorThrown) {
                console.error('Error fetching models:', textStatus, errorThrown);
            }
        });
    }

    // Function to populate dropdown with models
    function populateModelDropdown(models) {
        $.each(models, function(index, model) {
            modelSelect.append($('<option>', {
                value: model.modelId,
                text: model.modelId
            }));
        });
    }

    // Fetch models when the document is ready
    fetchModels();

    
}