const AiDialModelsButton = {
    delimiters: ['[[', ']]'],
    props: ['error', 'pluginName', 'is_loading_models', 'body_data', 'models'],
    emits: ['handleError', 'update:error', 'update:models'],
    data() {
        return this.initialState()
    },
    computed: {
        test_connection_class() {
            if (400 <= this.status && this.status < 600) {
                return 'btn-warning'

            } else {
                return 'btn-secondary'
            }
        },
        show_error() {
            return (this.test_connection_class === 'btn-warning')
        },
        hasErrors() {
            return this.warnings.length > 0
        },
    },
    template: `
    <div>
        <button type="button" class="btn btn-sm mt-3"
                @click="loadModels"
                :class="[{disabled: is_loading_models, updating: is_loading_models, 'is-invalid': error}, test_connection_class]"
        >
            Load models
        </button>
        <button type="button" class="btn btn-sm mt-3"
                @click="clearModels"
                class="btn-secondary"
        >
            Clear models
        </button>
        <div class="invalid-feedback" v-if="show_error">[[ error ]]</div>
        <div v-if="is_models_loaded" class="mr-2 cell-input" style="min-width: 250px">
            <div>
                <multiselect-dropdown
                    placeholder="Select models"
                    maxHeight="300"
                    v-model="selected_models"
                    :return_key="null"
                    :list_items="allModels.map(model => {
                        model.name = model.id
                        return model
                    })"
                ></multiselect-dropdown>
            </div>
        </div>
        <p class="font-h5 font-semibold mt-3">Add model names:</p>
        <div class="input-group d-flex mt-1">
            <div class="custom-input flex-grow-1">
                <input type="text" placeholder="Model name" class="form-control form-control-alternative"
                    v-model="model"
                    :class="{ 'is-invalid': hasErrors }"
            >
            </div>
            <div class="flex-grow-1 ml-2">
                <Multiselect-Dropdown
                        placeholder="Capabilities"
                        instance_name="capabilities_select"
                        :list_items='["completion", "chat_completion", "embeddings"]'
                        :pre_selected_indexes='[1]'
                        v-model:modelValue="capabilities"
                        container_class="metric_select bootstrap-select__b"
                ></Multiselect-Dropdown>
            </div>
            <button class="btn btn-lg btn-secondary ml-2" type="button"
                @click="handleAdd"
                :disabled="model === ''"
                :class="{ 'btn-danger': hasErrors }"
            >
                Add
            </button>
            <div class="invalid-feedback d-block" v-for="warning in warnings">[[ warning ]]</div>
        </div>
    </div>
    `,
    watch: {
        is_loading_models(newState, oldState) {
            if (newState) {
                this.status = 0
            }
        },
        selected_models(newState, oldState) {
            if (newState) {
                this.$emit('update:models', newState)
            }
        }
    },
    methods: {
        clear() {
            Object.assign(this.$data, this.initialState())
        },
        async loadModels() {
            this.is_loading_models = true
            this.loadModelsAPI(this.pluginName, this.body_data).then(res => {
                this.allModels = res;
                // this.selected_models = res.filter(model => this.models.includes(model.model)).map(model => model.model);
                // this.selected_models = res.filter(
                //     model => { for (const m of this.models) {
                //         if (m.id === model.id) {
                //             return true
                //         }
                //     }
                // });
                // debugger
            })
        },
        clearModels() {
            this.selected_models = []
        },
        async loadModelsAPI(integration_name, settings) {
            const api_url = V.build_api_url(integration_name, 'models')
            const response = await fetch(`${api_url}/${getSelectedProjectId()}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(settings)
            })
            this.status = response.status
            if (!response.ok) {
                this.is_loading_models = false
                this.$emit('handleError', response)
                return []
            } else {
                this.is_models_loaded = true
                this.is_loading_models = false
                return response.json();
            }
        },
        validateUniqueness(model_name) {
            return this.models.find(e => e.id.toLowerCase() === model_name.toLowerCase()) === undefined
        },
        add(model_name) {
            if (model_name === '') return;
            if (!this.validateUniqueness(model_name)) {
                this.warnings.push(`Model ${model_name} is already added`)
                return;
            }
            if (this.capabilities.length === 0) {
                this.warnings.push(`Please select capabilities for model ${model_name}`)
                return;
            }
            const model = {
                id: model_name,
                name: model_name,
                capabilities: {
                    completion: this.capabilities.includes('completion'),
                    chat_completion: this.capabilities.includes('chat_completion'),
                    embeddings: this.capabilities.includes('embeddings'),
                }
            }
            this.selected_models = [...this.models, model];
        },
        handleAdd() {
            this.warnings = []
            this.model.split(',').forEach(i => {
                this.add(i.trim().toLowerCase())
            })
            if (!this.hasErrors) {
                this.model = ''
            }
        },
        initialState: () => ({
            status: 0,
            model: 'gpt-35-turbo',
            allModels: [],
            is_loading_models: false,
            is_models_loaded: false,
            selected_models: [],
            capabilities: [],
            warnings: [],
        })
    }
}
