-- Copyright (C) PerfGao

local ffi = require 'ffi'

ffi.cdef[[
    void *lgbm_load(char *model_filename);
    void lgbm_free(void *lightgbm);
    double lgbm_predict(void *lightgbm, double *data, int data_size);
]]

local type = type

local lightgbm_lib = nil
local lightgbm_load = nil
local lightgbm_predict = nil
local lightgbm_free = nil

local function load_shared_obj(so_path)
    if lightgbm_lib ~= nil then
        return lightgbm_lib
    end

    if type(so_path) ~= 'string' then
        return nil, "invalid so path"
    end

    lightgbm_lib = ffi.load(so_path)

    lightgbm_load    = lightgbm_lib.lgbm_load
    lightgbm_predict = lightgbm_lib.lgbm_predict
    lightgbm_free    = lightgbm_lib.lgbm_free

    return lightgbm_lib
end

local function create_lgbm(model_filename, config_filename)
    if lightgbm_lib == nil then
        return nil, 'maybe not load lib_lightgbm.so'
    end

    if type(model_filename) ~= 'string' then
        return nil, 'invalid filename'
    end

    local c_model = ffi.new("char[?]", #model_filename + 1)

    ffi.copy(c_model, model_filename)

    local obj = lightgbm_load(c_model)
    if not obj or obj == ffi.NULL then
        return nil, "load model failed!"
    end

    return ffi.gc(obj, lightgbm_free)
end

local function predict(obj, ori_data)
    if not ori_data then
        return nil
    end

    local data = ffi.new("double [?]", #ori_data)

    for i = 1, #ori_data, 1 do
        data[i - 1] = ori_data[i]
    end

    return lightgbm_predict(obj, data, #ori_data)
end

local _M = {
    _VERSION = '0.01',
    loadlib = load_shared_obj,
    create = create_lgbm,
    predict = predict,
}

return _M
