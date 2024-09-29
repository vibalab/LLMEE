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
        $('.answer-area').append(`<p>Predicted Output: <span class='predicted'>${$('.query')[1].value}</span></p>`);

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
                    const modelName = model.split('/')[1]
                    const card = $('<div>', {
                        class : `model-card ${modelName}`
                    });

                    const cardHeader = $('<div>', {
                        text: modelName,
                        class : `card-header ${modelName}`
                    });

                    const cardBody = $('<div>', {
                        class : `card-body ${modelName}`
                    });

                    cardBody.html(`
                        <p>Input Text: <span class=input-seq>${inputSentence}</span></p>
                        <p class=generated-text>Generated Text: ${data.results[model]['generated_text']}</p>
                        <div class=scores>
                            <span>ROUGE-1 Score: ${data.results[model]['rouge_score']['rouge1'][2].toFixed(3)}</span>
                            <span>ROUGE-2 Score: ${data.results[model]['rouge_score']['rouge2'][2].toFixed(3)}</span>
                            <span>ROUGE-L Score: ${data.results[model]['rouge_score']['rougeL'][2].toFixed(3)}</span>
                            <span>BLEURT Score: ${data.results[model]['bleurt_score'].toFixed(3)}</span>
                        </div>
                        `);

                    const metaData = data.results[model]['model_card'];
                    const content = $('<table>', {
                        class:`card-meta ${modelName}`
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

                    card.append(cardHeader);
                    card.append(cardBody);


                    $('.answer-area').append(card);
                    seqAttrColoring(data.results[model], modelName);
                }
            })
            headerEvent();
            setupWordClickHandler(data, model_list)
        })
    })

    function headerEvent(){
        $('.card-header').on('click', function(e){
            const isShirinked = $(this).parent().css('max-height') == "105px";
    
            console.log(isShirinked);
            
            if(isShirinked){
                // $(this).parent().css('flex-basis', "");
                $(this).parent().css('max-height', "5000px");
                // $(this).next().css('overflow-y', "scroll");
                // $(this).next().css('height', "auto");
            } else{
                // $(this).parent().css('flex-basis', "100px");
                $(this).parent().css('max-height', "105px");
                // $(this).next().css('overflow-y', "hidden");
                // $(this).next().css('height', "100px");
            }        
        })
    }

    // 1. 단어 클릭 핸들러 설정 함수
    function setupWordClickHandler(data, model_list) {
        // 문장 선택
        const sentenceSpan = $('body > div.container > div.main.item > div.result > div > p > span');
        const sentenceText = sentenceSpan.text();

        // 2. 문장을 단어로 분할하고 클릭 가능하게 설정
        const wordsWithSpaces = sentenceText.match(/\S+\s*/g) || [];

        let wordIndex = 0;
        const newHTML = wordsWithSpaces.map(word => {
            const trimmedWord = word.trim();
            const spaceAfter = word.slice(trimmedWord.length);
            const html = `<span class="word" data-index="${wordIndex}">${trimmedWord}</span>${spaceAfter}`;
            wordIndex++;
            return html;
        }).join('');

        // 기존 문장을 새로운 HTML로 대체
        sentenceSpan.html(newHTML);
        $('span.word').css('cursor', 'pointer');

        // 각 모델의 문장도 단어별로 감싸기
        wrapTokensForModels(data, model_list);

        // 3. 단어 클릭 이벤트 설정
        $('span.word').off('click').on('click', function() {
            const index = parseInt($(this).attr('data-index'));

            $('span.word').css('background-color', ''); // 이전에 적용된 스타일 초기화
            $(this).css('background-color', '#f0f0f0'); // 옅은 회색 음영

            $('p.value-container').remove();
            $('.value-row').remove();

            // 4. 모델 리스트 순회
            model_list.forEach(function(modelName) {
                const modelData = data.results[modelName];
                const tokenAttr = modelData['token_attr'];
                // 4. token_attr 존재 여부 확인
                if (!tokenAttr) {
                    // 다음 모델로 넘어감
                    return;
                }
                // 5. token_attr를 사용하여 배경색 적용
                performBackgroundColoring(modelName, tokenAttr[index], index, modelData['input_tokens']);
            });
        });
    }

    function wrapTokensInSpans(model, modelName) {
        const spanElements = $(`.card-body[class*='${modelName}'] span.input-seq`);
        const input_seq = spanElements[0].innerText;
        const input_tokens = model.input_tokens;
    
        // 토큰을 감싸는 새로운 HTML을 생성
        let tokenizedHTML = '';
        let currentIndex = 0;
    
        input_tokens.forEach((token) => {
            // 특수 문자를 이스케이프하여 정규식을 안전하게 만듭니다.
            const escapedToken = token.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');
    
            // 현재 인덱스부터 토큰의 위치를 찾습니다.
            const tokenPosition = input_seq.indexOf(token, currentIndex);
    
            if (tokenPosition !== -1) {
                // 토큰 이전의 텍스트를 추가 (공백이나 구두점 등)
                if (tokenPosition > currentIndex) {
                    const beforeToken = input_seq.substring(currentIndex, tokenPosition);
                    tokenizedHTML += beforeToken; // 토큰 이전의 텍스트는 감싸지 않음
                }
    
                // 토큰을 `<span>`으로 감쌉니다.
                tokenizedHTML += `<span class="token">${token}</span>`;
    
                // 현재 인덱스를 업데이트
                currentIndex = tokenPosition + token.length;
            }
        });
    
        // 마지막 토큰 이후의 남은 텍스트를 추가
        if (currentIndex < input_seq.length) {
            const remainingText = input_seq.substring(currentIndex);
            tokenizedHTML += remainingText;
        }
    
        // 기존의 내용물을 새로운 HTML로 교체
        spanElements[0].innerHTML = tokenizedHTML;
    }

    // 2. 각 모델의 문장에 단어별로 span 감싸기
    function wrapTokensForModels(data, model_list) {
        model_list.forEach(function(modelName) {
            const sentenceSpan = $(`.card-body[class*='${modelName}'] span.input-seq`);
            const sentenceText = sentenceSpan.text();

            const wordsWithSpaces = sentenceText.match(/\S+\s*/g) || [];

            let wordIndex = 0;
            const newHTML = wordsWithSpaces.map(word => {
                const trimmedWord = word.trim();
                const spaceAfter = word.slice(trimmedWord.length);
                const html = `<span class="token" data-index="${wordIndex}">${trimmedWord}</span>${spaceAfter}`;
                wordIndex++;
                return html;
            }).join('');

            sentenceSpan.html(newHTML);
        });
    }

    // 3. 배경색 적용 함수
    function performBackgroundColoring(modelName, tokenAttr, selected_index, input_tokens) {
        const modelName_split = modelName.split('/')[1]
        const sentenceSpan = $(`.card-body[class*='${modelName_split}'] span.input-seq`);
        const tokenSpans = sentenceSpan.find('span.token');

        // seq_attr와 동일하게 min, max 값 계산
        const flatAttr = Array.isArray(tokenAttr[0]) ? tokenAttr.flat() : tokenAttr;
        const minAttr = Math.min(...flatAttr);
        const maxAttr = Math.max(...flatAttr);

        const valueRow = $('<p class="value-row"></p>').css({
            'margin-top': '0px',  // 두 p 사이 간격 줄이기
            'margin-bottom': '0px',  // 두 p 사이 간격 줄이기
            'padding-left': '130px',  // 'Input Text'의 영향으로 좌우 맞춤
            'font-size': '0.8em'
        });
        // 각 토큰에 배경색 적용
        tokenSpans.each(function(index) {
            const value = Array.isArray(tokenAttr[index]) ? tokenAttr[index][0] : tokenAttr[index];
            if(input_tokens[index] != ''){
                const color = mapAttrColor(value, minAttr, maxAttr);
                $(this).css('background-color', color);
                const roundedValue = value.toFixed(3);  // 소숫점 3자리로 반올림

                const $valueContainer = $('<span></span>').css({
                    'font-size': '0.8em',
                    'color': color,
                    'margin-right': '10px'  // 각 숫자 간 간격 추가
                }).text(roundedValue);

                valueRow.append($valueContainer);  // 숫자 추가
            }
        });
        sentenceSpan.after(valueRow);

    }

    function seqAttrColoring(model, modelName){  
        const input_tokens = model.input_tokens;
        const seq_attr = model.seq_attr;
    
        // 먼저 각 토큰을 `<span>`으로 감쌉니다.
        wrapTokensInSpans(model, modelName);
    
        // 토큰별로 감싸진 `<span>` 요소들을 선택합니다.
        const tokenSpans = $(`.card-body[class*='${modelName}'] span.input-seq span.token`);
    
        // `seq_attr`의 최소값과 최대값을 계산
        const minAttr = Math.min(...seq_attr);
        const maxAttr = Math.max(...seq_attr);
    
        // 각 토큰에 배경색을 적용
        tokenSpans.each(function(index) {
            const token = $(this).text();
            const score = seq_attr[index];
    
            const color = mapAttrColor(score, minAttr, maxAttr);
    
            $(this).css('background-color', color);
        });
    }
    
    // Function to get the color based on seq_attr value
    function mapAttrColor(value, minAttr, maxAttr) {
        let color;
        let alpha;

        if (value < 0) {
            // 음수 값 매핑: #1E88E5
            const normalized = (value - 0) / (minAttr - 0);
            alpha = normalized;
            color = hexToRgba('#1E88E5', alpha);
        } else if (value > 0) {
            // 양수 값 매핑: #FF0D57
            const normalized = value / maxAttr;
            alpha = normalized;
            color = hexToRgba('#FF0D57', alpha);
        } else {
            // 값이 0인 경우
            alpha = 0;
            color = 'rgba(0, 0, 0, 0)';
        }
        return color;
    }

    // Helper function to convert hex color code to RGBA with alpha
    function hexToRgba(hex, alpha) {
        hex = hex.replace('#', '');

        if (hex.length === 3) {
            hex = hex.split('').map(function (hexChar) {
                return hexChar + hexChar;
            }).join('');
        }

        var r = parseInt(hex.substring(0,2), 16);
        var g = parseInt(hex.substring(2,4), 16);
        var b = parseInt(hex.substring(4,6), 16);

        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }
}